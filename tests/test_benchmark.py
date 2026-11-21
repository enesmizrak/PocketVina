"""Repository-only benchmark contracts; public datasets and GPUs are never needed."""

import copy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "benchmark").is_dir() and (ROOT / "PKG-INFO").is_file():
    pytest.skip("an sdist deliberately excludes benchmark/", allow_module_level=True)

from benchmark.protocol import (  # noqa: E402
    Outcome,
    Pose,
    digest,
    load_protocol,
    pose_metrics,
    requested_ids,
    summarize,
    validate,
)


@pytest.fixture
def protocol():
    return load_protocol(ROOT / "benchmark/protocols/smoke.json")


def test_protocol_bundled_counts():
    counts = {}
    for path in (ROOT / "benchmark/protocols").glob("*.json"):
        p = load_protocol(path)
        if p["availability"] != "unavailable":
            counts[p["name"]] = len(requested_ids(p))
        else:
            with pytest.raises(ValueError, match="split|Authoritative"):
                requested_ids(p)
    assert counts == {
        "pdbbind-bundled357": 357,
        "pdbbind-unseen-bundled144": 144,
        "posebusters-bundled427": 427,
        "astex-bundled85": 85,
        "smoke": 3,
    }


def test_protocol_hash_count_duplicates_missing_exclusions(protocol, tmp_path):
    from benchmark.protocol import json_digest

    for ids, count, match in [(["a", "a"], 2, "duplicate"), (["a"], 2, "count"), ([""], 1, "nonblank")]:
        p = copy.deepcopy(protocol)
        p["cohort"].update(case_ids=ids, expected_count=count, split_sha256=json_digest(ids))
        with pytest.raises(ValueError, match=match):
            requested_ids(p)
    protocol["cohort"]["exclusions"] = {"missing": "documented"}
    with pytest.raises(ValueError, match="missing"):
        requested_ids(protocol)
    path = tmp_path / "split"
    path.write_text("a\n\nb\n")
    protocol.update(adapter="pdbbind", availability="external")
    protocol["cohort"].update(split_path=str(path), split_sha256=digest(path), expected_count=2, exclusions={})
    assert requested_ids(validate(protocol)) == ["a", "b"]
    path.write_text("a\nb\nc\n")
    with pytest.raises(ValueError, match="SHA256"):
        requested_ids(protocol)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("adapter", "unknown"),
        ("seeds", []),
        ("seeds", [42, 42]),
        ("seeds", [-1]),
        ("top_k", 0),
        ("preparation", "implicit"),
        ("posebusters", "yes"),
    ],
)
def test_protocol_rejects_invalid(protocol, field, value):
    protocol[field] = value
    with pytest.raises(ValueError):
        validate(protocol)


def outcome(case_id, rmsd=None, status="ok", seed=42):
    result = Outcome(case_id, seed, "run", "legacy-smiles", status=status)
    if rmsd is not None:
        result.poses = [Pose("pose1", -7, rmsd)]
        result.produced_pose_count = 1
    return result


def test_accounting_full_denominator():
    summary = summarize([outcome("accurate", 1), outcome("inaccurate", 4), outcome("conversion", status="failed")])
    assert summary["oracle_success_le2"] == {"count": 1, "denominator": 3, "percent": pytest.approx(100 / 3)}
    assert summary["evaluated_case_seeds"] == 2
    assert summary["failed_case_seeds"] == 1
    assert summary["oracle_median_rmsd_evaluated_only"] == 2.5


def test_accounting_empty_allfailed_excluded():
    assert summarize([])["oracle_median_rmsd_evaluated_only"] is None
    summary = summarize([outcome("failed", status="failed"), outcome("excluded", status="excluded")])
    assert summary["oracle_success_le2"] == {"count": 0, "denominator": 1, "percent": 0}
    assert summary["requested_case_seeds"] == 2
    assert summary["excluded_case_seeds"] == 1
    assert summary["oracle_median_rmsd_evaluated_only"] is None
    with pytest.raises(ValueError, match="duplicate"):
        summarize([outcome("a"), outcome("a")])
    with pytest.raises(ValueError, match="unfinished"):
        summarize([outcome("a", status="pending")])


@pytest.fixture
def real_protocol(protocol, tmp_path):
    split = tmp_path / "split"
    split.write_text("case-a\ncase-b\n")
    protocol.update(adapter="pdbbind", availability="external", dataset_root=str(tmp_path / "dataset"))
    protocol["cohort"] = {
        "split_path": str(split),
        "split_sha256": digest(split),
        "expected_count": 2,
        "exclusions": {},
    }
    return protocol


def fake_inputs(protocol):
    from benchmark.adapters import PDBBindAdapter

    for case_id in requested_ids(protocol):
        files = PDBBindAdapter().locate(protocol["dataset_root"], case_id)
        files.protein.parent.mkdir(parents=True)
        for path in files.paths("legacy-smiles").values():
            path.write_text("fixture-input\n")


@pytest.mark.parametrize("recipe", ["reference-sdf-topology-historical", "legacy-smiles", "standard-smiles-seeded"])
def test_fakeengine_recipe_seed_forwarding_and_two_seed_ledger(real_protocol, tmp_path, recipe):
    from benchmark.run import execute

    fake_inputs(real_protocol)
    real_protocol["preparation"] = recipe
    calls = []

    def prepare(files, case_id, recipe, seed, work):
        calls.append((case_id, recipe, seed))
        assert not list(work.glob("batch*"))
        return {"recipe": recipe, "engine_seed": seed, "prepared_input_sha256": digest(files.reference)}

    def dock(files, protocol, seed, work):
        assert protocol["preparation"] == recipe and seed in (42, 43)
        return [files.protein]

    def evaluate(files, outputs, work, result, use_pb):
        result.poses = [Pose("a", -8, 1.5), Pose("b", -9, 3)]
        result.produced_pose_count = 2

    report = execute(
        real_protocol, tmp_path / "reports", prepare=prepare, dock=dock, evaluate=evaluate, identity={"fake": True}
    )
    assert len(calls) == len(report["outcomes"]) == 4
    assert {c[1] for c in calls} == {recipe}
    assert {c[2] for c in calls} == {42, 43}
    assert report["summary"]["oracle_success_le2"]["count"] == 4
    assert report["summary"]["top1_success_le2"]["count"] == 0
    assert set(report["summary_by_seed"]) == {"42", "43"}
    for row in report["outcomes"]:
        assert row["status"] == "ok" and row["evaluated_pose_count"] == row["produced_pose_count"] == 2
        assert len(row["input_digests"]) >= 2
        assert row["stages"]["preparation"]["engine_seed"] == row["seed"]
    saved = json.loads((tmp_path / "reports/report.json").read_text())
    assert saved == report
    import csv

    assert len(list(csv.DictReader((tmp_path / "reports/outcomes.csv").open()))) == 4
    with pytest.raises(FileExistsError):
        execute(real_protocol, tmp_path / "reports", identity={"fake": True})


def test_no_import_time_work_help_dryrun_and_run_label():
    import subprocess
    import sys

    code = """
import sys
sys.argv = ['benchmark', '--unknown-import-must-not-parse']
import benchmark.protocol, benchmark.run, benchmark.adapters
import benchmark.pdbbind.main, benchmark.astex.main, benchmark.posebuster.main, benchmark.dockgen.main
assert not any(x in sys.modules for x in ('rdkit', 'networkx', 'spyrmsd', 'pocketvina'))
"""
    result = subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    help_result = subprocess.run(
        [sys.executable, "-S", "-m", "benchmark.run", "--help"], cwd=ROOT, capture_output=True, text=True
    )
    assert help_result.returncode == 0 and "--seeds" in help_result.stdout
    reports = []
    for label in ("first", "second"):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-m",
                "benchmark.run",
                "--protocol",
                "benchmark/protocols/smoke.json",
                "--dry-run",
                "--run-label",
                label,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        reports.append(json.loads(result.stdout))
    assert reports[0]["work"] == reports[1]["work"]
    assert reports[0]["expected_docking_jobs"] == 6


def test_synthetic_execution_rejected(protocol, tmp_path):
    from benchmark.run import execute

    with pytest.raises(ValueError, match="synthetic smoke"):
        execute(protocol, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_oracle_top1_topk_missingpb_and_stable_ties():
    result = outcome("a")
    result.poses = [Pose("z", -8, 0.5, True), Pose("a", -9, 5), Pose("b", -9, 1, False)]
    metrics = pose_metrics(result, top_k=2)
    assert metrics["oracle_min_rmsd"] == 0.5
    assert metrics["oracle_success_le2"] is True
    assert metrics["top1_success_le2"] is False
    assert metrics["topk_success_le2"] is True
    assert metrics["topk_pb_valid_success_le2"] is False
    assert metrics["pb_known_poses"] == 2 and metrics["pb_total_poses"] == 3
    assert metrics["top1_pose_id"] == "a"
    result.poses.reverse()
    assert pose_metrics(result, top_k=2) == metrics
    assert pose_metrics(result, top_k=3)["topk_pb_valid_success_le2"] is True
    result.poses = [Pose("a", -9, 1)]
    assert pose_metrics(result)["top1_success_le2"] is True
    assert pose_metrics(result)["top1_pb_valid_success_le2"] is False
    result.status = "failed"
    assert pose_metrics(result)["oracle_success_le2"] is False
    assert pose_metrics(result)["oracle_min_rmsd"] == 1


@pytest.mark.parametrize("stage", ["inputs", "preparation", "docking", "evaluation"])
def test_stagefailure_reports_all_requested(real_protocol, tmp_path, stage):
    from benchmark.run import execute

    fake_inputs(real_protocol)
    if stage == "inputs":
        from benchmark.adapters import PDBBindAdapter

        for case in requested_ids(real_protocol):
            PDBBindAdapter().locate(real_protocol["dataset_root"], case).reference.unlink()

    def fail(*args):
        raise ValueError("intentional failure")

    def prepare(*args):
        return {}

    def dock(*args):
        return []

    def evaluate(*args):
        return None

    functions = {
        "prepare": fail if stage == "preparation" else prepare,
        "dock": fail if stage == "docking" else dock,
        "evaluate": fail if stage == "evaluation" else evaluate,
    }
    report = execute(real_protocol, tmp_path / "failed", identity={"fake": True}, **functions)
    assert report["summary"]["failed_case_seeds"] == 4
    assert report["summary"]["oracle_success_le2"] == {"count": 0, "denominator": 4, "percent": 0}
    assert all(o["stage"] == stage and o["status"] == "failed" and o["error"] for o in report["outcomes"])
    assert (tmp_path / "failed/outcomes.csv").is_file()
    assert (tmp_path / "failed/poses.csv").is_file()


def test_fixture_conversion_preserves_row_identity_and_symmetry(tmp_path):
    from benchmark.adapters import CaseFiles, evaluate_outputs, load_rmsd_mol, symmetry_rmsd

    pytest.importorskip("spyrmsd")
    pytest.importorskip("networkx")
    import subprocess

    from rdkit import Chem
    from spyrmsd.molecule import Molecule
    from spyrmsd.rmsd import rmsdwrapper

    from pocketvina.ligands import find_obabel
    from pocketvina.pdbqt import check_modes

    fixture = ROOT / "tests/data/legacy_run/batch_1_out/AF-Q96LA5-F1-model_v4-pocket1-molecule2_cleaned_out.pdbqt"
    raw = fixture.read_bytes()
    first = check_modes(raw)[0]
    first_pdbqt, reference = tmp_path / "first.pdbqt", tmp_path / "reference.sdf"
    first_pdbqt.write_bytes(raw[first.raw_offset : first.raw_offset + first.raw_length])
    subprocess.run([find_obabel(), str(first_pdbqt), "-osdf", "-O", str(reference)], check=True, capture_output=True)
    files = CaseFiles(tmp_path / "unused-protein.pdb", reference, tmp_path / "unused-smiles")
    result = outcome("fixture")
    evaluate_outputs(files, [fixture], tmp_path, result, False)
    assert len(result.poses) == result.produced_pose_count == len(check_modes(raw))
    for pose, mode in zip(result.poses, check_modes(raw)):
        assert pose.affinity == mode.affinity
        assert pose.pose_id.endswith(f"mode{mode.mode:03d}")
        assert pose.error is None and pose.rmsd is not None
        candidate = tmp_path / "evaluation" / f"{pose.pose_id}.sdf"
        ref, cand = Molecule.from_rdkit(load_rmsd_mol(reference)), Molecule.from_rdkit(load_rmsd_mol(candidate))
        ref.strip()
        cand.strip()
        assert pose.rmsd == pytest.approx(float(rmsdwrapper(ref, cand, symmetry=True, minimize=False)[0]))
    assert result.poses[0].rmsd == pytest.approx(0)
    mol = load_rmsd_mol(reference)
    conformer = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        xyz = conformer.GetAtomPosition(i)
        conformer.SetAtomPosition(i, (xyz.x + 3, xyz.y, xyz.z))
    shifted = tmp_path / "shifted.sdf"
    with Chem.SDWriter(str(shifted)) as writer:
        writer.write(mol)
    assert symmetry_rmsd(reference, shifted) == pytest.approx(3, abs=0.001)
    wrong = tmp_path / "wrong.sdf"
    with Chem.SDWriter(str(wrong)) as writer:
        writer.write(Chem.MolFromSmiles("C"))
    with pytest.raises(ValueError, match="identity mismatch"):
        symmetry_rmsd(reference, wrong)


def test_real_recipe_selection_and_fake_engine_seed(tmp_path, monkeypatch, fake_engine_env):
    pytest.importorskip("networkx")
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from benchmark import adapters
    from pocketvina import main as pv_main
    from test_pipeline import FIXTURE_RECEPTOR, write_pockets

    reference, smiles = tmp_path / "reference.sdf", tmp_path / "ligands.txt"
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    with Chem.SDWriter(str(reference)) as writer:
        writer.write(mol)
    smiles.write_text("molecule_name,smiles\nethanol,CCO\n")
    files = adapters.CaseFiles(FIXTURE_RECEPTOR, reference, smiles)
    protocol = load_protocol(ROOT / "benchmark/protocols/smoke.json")
    protocol["settings"]["top_pockets"] = 1

    def p2rank(processor, pdb_file):
        write_pockets(Path(processor.pockets_dir) / f"{Path(pdb_file).stem}.csv", n_pockets=1)
        return True

    monkeypatch.setattr(pv_main.P2RankProcessor, "run_p2rank", p2rank)
    for i, recipe in enumerate(("reference-sdf-topology-historical", "legacy-smiles", "standard-smiles-seeded")):
        work = tmp_path / recipe
        work.mkdir()
        seed = 42 + i
        details = adapters.prepare_case(files, "ethanol", recipe, seed, work)
        assert details["recipe"] == recipe and details["engine_seed"] == seed
        assert details["prepared_input_sha256"] == digest(work / "batch_1/ethanol_cleaned.pdbqt")
        protocol["preparation"] = recipe
        outputs = adapters.dock_case(files, protocol, seed, work)
        assert outputs
    calls = fake_engine_env.calls()
    assert [c["options"]["seed"] for c in calls] == [42, 43, 44]
    assert all(c["returncode"] == 0 for c in calls)


def test_partial_docking_and_missing_pb_are_separate(real_protocol, tmp_path):
    from benchmark.adapters import DockingOutputs
    from benchmark.run import execute

    fake_inputs(real_protocol)
    real_protocol["posebusters"] = True

    def prepare(*args):
        return {}

    def evaluate(files, outputs, work, result, use_pb):
        assert use_pb
        result.poses = [Pose("sample", -7, 1, pb_error="unavailable check")]
        result.produced_pose_count = 1

    report = execute(
        real_protocol,
        tmp_path / "unknown-pb",
        prepare=prepare,
        dock=lambda *a: [],
        evaluate=evaluate,
        identity={"fake": True},
    )
    assert report["summary"]["oracle_success_le2"]["count"] == 4
    assert report["summary"]["top1_pb_valid_success_le2"]["count"] == 0
    assert report["summary"]["pb_known_poses"] == 0
    assert all(row["stages"]["posebusters"]["status"] == "incomplete" for row in report["outcomes"])
    report = execute(
        real_protocol,
        tmp_path / "partial",
        prepare=prepare,
        dock=lambda *a: DockingOutputs([], ["one requested pocket failed"]),
        evaluate=evaluate,
        identity={"fake": True},
    )
    assert report["summary"]["oracle_success_le2"]["count"] == 0
    assert report["summary"]["evaluated_case_seeds"] == 4
    assert report["summary"]["failed_case_seeds"] == 4
    assert all(row["stage"] == "docking" and row["produced_pose_count"] == 1 for row in report["outcomes"])


def test_atomic_report_interruption(tmp_path):
    from benchmark.run import atomic_write

    target = tmp_path / "report.json"
    target.write_text("original")

    def interrupted(path):
        path.write_text("partial")
        raise RuntimeError("interrupted write")

    with pytest.raises(RuntimeError, match="interrupted"):
        atomic_write(target, interrupted)
    assert target.read_text() == "original"
    assert list(tmp_path.iterdir()) == [target]


def test_empty_protocol_and_excluded_ledger(real_protocol, tmp_path):
    from benchmark.run import execute

    real_protocol["cohort"]["exclusions"] = {"case-a": "documented exclusion", "case-b": "documented exclusion"}
    report = execute(real_protocol, tmp_path / "excluded", identity={"fake": True})
    assert report["summary"]["excluded_case_seeds"] == 4
    assert report["summary"]["eligible_case_seeds"] == 0
    assert all(row["status"] == "excluded" for row in report["outcomes"])
    split = Path(real_protocol["cohort"]["split_path"])
    split.write_text("")
    real_protocol["cohort"].update(split_sha256=digest(split), expected_count=0, exclusions={})
    report = execute(real_protocol, tmp_path / "empty", identity={"fake": True})
    assert report["outcomes"] == []
    assert report["summary"]["requested_case_seeds"] == 0
