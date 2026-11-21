"""Run repository benchmark protocols. Help and dry-run require only the standard library."""

import argparse
import copy
import csv
import json
import os
import tempfile
from pathlib import Path

from benchmark.protocol import (
    CONVENTIONS,
    ROOT,
    Outcome,
    json_digest,
    load_protocol,
    requested_ids,
    summarize,
    validate,
)


def dry_run(protocol):
    ids = requested_ids(protocol)
    exclusions = protocol["cohort"]["exclusions"]
    return {
        "dry_run": True,
        "protocol": protocol,
        "protocol_sha256": json_digest(protocol),
        "requested_cases": ids,
        "excluded_cases": exclusions,
        "expected_case_seeds": len(ids) * len(protocol["seeds"]),
        "expected_docking_jobs": (len(ids) - len(exclusions)) * len(protocol["seeds"]),
        "work": [
            {
                "case_id": case,
                "seed": seed,
                "preparation": protocol["preparation"],
                "status": "excluded" if case in exclusions else "planned",
            }
            for seed in protocol["seeds"]
            for case in ids
        ],
        "conventions": CONVENTIONS,
    }


def atomic_write(path, write):
    """Replace a complete report file atomically; interrupted writes never truncate an old file."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    os.close(fd)
    try:
        write(Path(temporary))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_reports(directory, protocol, identity, outcomes):
    """The JSON ledger is authoritative; CSV contains one row per requested case/seed, even all-failed."""
    from benchmark.adapters import write_pose_csv

    top_k = protocol["top_k"]
    rows = [o.record(top_k) for o in outcomes]
    report = {
        "schema_version": 1,
        "protocol": protocol,
        "protocol_sha256": json_digest(protocol),
        "environment": identity,
        "conventions": CONVENTIONS,
        "outcomes": rows,
        "summary": summarize(outcomes, top_k),
        "summary_by_seed": {
            str(seed): summarize([o for o in outcomes if o.seed == seed], top_k) for seed in protocol["seeds"]
        },
    }
    directory = Path(directory)
    atomic_write(
        directory / "report.json",
        lambda p: p.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"),
    )

    def write_csv(path):
        fields = [
            "case_id",
            "seed",
            "run_label",
            "preparation",
            "status",
            "stage",
            "error",
            "produced_pose_count",
            "evaluated_pose_count",
            "metrics",
            "stages",
            "input_digests",
            "artifact_digests",
        ]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(row[key], sort_keys=True) if isinstance(row[key], dict) else row[key]
                        for key in fields
                    }
                )

    atomic_write(directory / "outcomes.csv", write_csv)
    atomic_write(directory / "poses.csv", lambda p: write_pose_csv(p, outcomes))
    return report


def execute(protocol, output, *, adapter=None, prepare=None, dock=None, evaluate=None, identity=None):
    """Complete one ledger record per requested case/seed; adapters/tools are injectable for CPU contracts."""
    validate(protocol)
    ids = requested_ids(protocol)
    if protocol["adapter"] == "synthetic":
        raise ValueError("manifest-only synthetic smoke cannot execute real docking; use --dry-run")
    if not protocol["dataset_root"]:
        raise ValueError("supply an explicit dataset_root in the protocol or --dataset-root")
    from benchmark import adapters

    adapter = adapter or adapters.DATASETS[protocol["adapter"]]()
    prepare = prepare or adapters.prepare_case
    dock = dock or adapters.dock_case
    evaluate = evaluate or adapters.evaluate_outputs
    identity = adapters.environment_identity() if identity is None else identity
    # Never reuse existing batches/poses: different seeds or recipes must start with fresh work directories.
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    outcomes = [
        Outcome(case, seed, protocol["run_label"], protocol["preparation"])
        for seed in protocol["seeds"]
        for case in ids
    ]
    exclusions = protocol["cohort"]["exclusions"]
    for outcome in outcomes:
        if outcome.case_id in exclusions:
            outcome.status, outcome.stage, outcome.error = "excluded", "exclusion", exclusions[outcome.case_id]
    # An initial machine-readable definition survives interruption before the first case is completed.
    atomic_write(
        output / "definition.json",
        lambda p: p.write_text(
            json.dumps(
                {"protocol": protocol, "protocol_sha256": json_digest(protocol), "environment": identity},
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        ),
    )
    for outcome in outcomes:
        if outcome.status == "excluded":
            continue
        work = output / f"seed-{outcome.seed}" / outcome.case_id
        work.mkdir(parents=True)
        try:
            files = adapter.locate(protocol["dataset_root"], outcome.case_id)
            for name, path in files.paths(protocol["preparation"]).items():
                from benchmark.protocol import digest

                outcome.input_digests[name] = {"path": str(path), "sha256": digest(path)}
            outcome.stages["inputs"] = {"status": "ok"}
            outcome.stage = "preparation"
            details = prepare(files, outcome.case_id, protocol["preparation"], outcome.seed, work)
            outcome.stages["preparation"] = dict(status="ok", **details)
            outcome.stage = "docking"
            outputs = dock(files, protocol, outcome.seed, work)
            docking_errors = getattr(outputs, "errors", [])
            outcome.stages["docking"] = {
                "status": "failed" if docking_errors else "ok",
                "output_files": len(outputs),
                "errors": docking_errors,
            }
            outcome.stage = "evaluation"
            evaluate(files, outputs, work, outcome, protocol["posebusters"])
            if any(p.error or p.rmsd is None for p in outcome.poses):
                raise ValueError("one or more poses failed conversion/evaluation; see pose errors")
            if not outcome.poses:
                raise ValueError("no evaluated poses")
            outcome.stages["evaluation"] = {"status": "ok"}
            if protocol["posebusters"]:
                unknown = [p.pose_id for p in outcome.poses if p.pb_valid is None]
                outcome.stages["posebusters"] = {
                    "status": "incomplete" if unknown else "ok",
                    "unknown_pose_ids": unknown,
                    "errors": [p.pb_error for p in outcome.poses if p.pb_error],
                }
            outcome.status = "ok"
            outcome.stage = "complete"
            if docking_errors:
                outcome.status, outcome.stage = "failed", "docking"
                outcome.error = "; ".join(docking_errors)
        except Exception as error:
            outcome.status = "failed"
            outcome.error = f"{type(error).__name__}: {error}"
            outcome.stages[outcome.stage] = {"status": "failed", "error": outcome.error}
        finally:
            outcome.artifact_digests = adapters.artifact_digests(work)
    return write_reports(output, protocol, identity, outcomes)


def parser(default_protocol=None):
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--protocol",
        required=default_protocol is None,
        default=default_protocol,
        help="Versioned JSON protocol (repository-relative split paths)",
    )
    result.add_argument("--dataset-root", help="Explicit processed dataset root")
    result.add_argument(
        "--results-dir", "--results_dir", dest="results_dir", help="New report directory (must not exist)"
    )
    result.add_argument("--dry-run", action="store_true", help="Validate cohort and list work; no scientific imports")
    result.add_argument(
        "--preparation", choices=("reference-sdf-topology-historical", "legacy-smiles", "standard-smiles-seeded")
    )
    result.add_argument("--seeds", type=int, nargs="+", help="Explicit engine seeds (run labels never choose seeds)")
    result.add_argument("--run-label", help="Artifact label; does not change seeds")
    result.add_argument("--run_number", type=int, help="Historical spelling: sets run_label only")
    for old in ("box_length", "vina_threads", "p2rank_threads", "p2rank_model"):
        result.add_argument(
            f"--{old}",
            type=str if old == "p2rank_model" else int,
            help="Historical spelling; overrides protocol setting",
        )
    return result


def main(argv=None, default_protocol=None, expected_adapter=None):
    arguments = parser(default_protocol).parse_args(argv)
    try:
        protocol = copy.deepcopy(load_protocol(arguments.protocol))
        if expected_adapter and protocol["adapter"] != expected_adapter:
            raise ValueError(f"wrapper requires adapter {expected_adapter}")
        for name in ("dataset_root", "preparation", "seeds", "run_label"):
            value = getattr(arguments, name)
            if value is not None:
                protocol[name] = value
        if arguments.run_number is not None:
            if arguments.run_label is not None:
                raise ValueError("choose --run-label or --run_number")
            protocol["run_label"] = f"run-{arguments.run_number}"
        for name in ("box_length", "vina_threads", "p2rank_threads", "p2rank_model"):
            value = getattr(arguments, name)
            if value is not None:
                protocol["settings"][name] = value
        validate(protocol)
        if arguments.dry_run:
            report = dry_run(protocol)
        else:
            if not arguments.results_dir:
                raise ValueError("--results-dir is required for execution")
            report = execute(protocol, arguments.results_dir)
        print(json.dumps(report if arguments.dry_run else report["summary"], indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser(default_protocol).error(str(error))


def legacy_main(dataset, argv=None):
    defaults = {
        "pdbbind": "pdbbind-bundled357",
        "astex": "astex-bundled85",
        "posebuster": "posebusters-bundled427",
        "dockgen": "dockgen-external-template",
    }
    return main(argv, str(ROOT / "benchmark/protocols" / f"{defaults[dataset]}.json"), expected_adapter=dataset)


if __name__ == "__main__":
    raise SystemExit(main())
