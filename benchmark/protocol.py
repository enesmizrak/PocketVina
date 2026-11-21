"""Versioned experiment definitions and failure-aware accounting (standard library only)."""

import hashlib
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
RECIPES = ("reference-sdf-topology-historical", "legacy-smiles", "standard-smiles-seeded")
ADAPTERS = ("pdbbind", "posebuster", "astex", "dockgen", "synthetic")
CONVENTIONS = {
    "rmsd": "spyrmsd rmsdwrapper(symmetry=True, minimize=False); no alignment; heavy atoms after strip",
    "identity": "equal heavy-atom element counts and bond-graph isomorphism required before RMSD",
    "ranking": "ascending affinity, then pose_id lexical; no reference RMSD in selection",
    "threshold_angstrom": 2.0,
    "missing_posebusters": "unknown, never valid",
}


def digest(path):
    """SHA256 of a file's bytes, streaming large input/output files."""
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def check_ids(ids):
    if not isinstance(ids, list) or any(
        not isinstance(i, str) or not re.fullmatch(r"[\w.-]+", i) or i in (".", "..") for i in ids
    ):
        raise ValueError("case IDs must be nonblank, safe single path components")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case IDs in requested cohort")
    return ids


def _positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate(protocol):
    """Validate definitions without reading datasets or importing scientific tools."""
    required = {
        "schema_version",
        "name",
        "adapter",
        "availability",
        "dataset_root",
        "cohort",
        "seeds",
        "preparation",
        "settings",
        "run_label",
        "top_k",
        "posebusters",
    }
    if not required <= protocol.keys():
        raise ValueError(f"missing protocol fields: {sorted(required - protocol.keys())}")
    if protocol["schema_version"] != 1 or protocol["adapter"] not in ADAPTERS:
        raise ValueError("unsupported protocol version or adapter")
    if protocol["availability"] not in ("bundled-subset", "external", "synthetic", "unavailable"):
        raise ValueError("invalid availability")
    for name in ("name", "run_label"):
        check_ids([protocol[name]])
    if protocol["preparation"] not in RECIPES:
        raise ValueError("unknown preparation recipe")
    if protocol["dataset_root"] is not None and not isinstance(protocol["dataset_root"], str):
        raise ValueError("dataset_root must be a path or null (CLI required)")
    seeds = protocol["seeds"]
    if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a nonempty unique list")
    for seed in seeds:
        _positive_int(seed, "seed", 0)
        if seed > 2147483647:
            raise ValueError("seed exceeds engine signed integer range")
    _positive_int(protocol["top_k"], "top_k")
    if type(protocol["posebusters"]) is not bool:
        raise ValueError("posebusters must be boolean")
    settings = protocol["settings"]
    if set(settings) != {"box_length", "vina_threads", "p2rank_threads", "p2rank_model", "top_pockets"}:
        raise ValueError("settings must explicitly specify box, threads, model and pocket selection")
    for name in ("box_length", "vina_threads", "p2rank_threads"):
        _positive_int(settings[name], name, 1000 if name == "vina_threads" else 1)
    if settings["top_pockets"] is not None:
        _positive_int(settings["top_pockets"], "top_pockets")
    if not isinstance(settings["p2rank_model"], str) or not settings["p2rank_model"]:
        raise ValueError("p2rank_model must be explicit")
    cohort = protocol["cohort"]
    if not {"expected_count", "split_path", "split_sha256", "exclusions"} <= cohort.keys():
        raise ValueError("missing cohort definition")
    _positive_int(cohort["expected_count"], "expected_count", 0)
    if not isinstance(cohort["exclusions"], dict) or any(
        not isinstance(reason, str) or not reason.strip() for reason in cohort["exclusions"].values()
    ):
        raise ValueError("exclusions require documented reasons")
    check_ids(list(cohort["exclusions"]))
    if protocol["availability"] == "unavailable":
        if not protocol.get("unavailable_reason"):
            raise ValueError("unavailable protocol requires a reason")
        return protocol
    if protocol["adapter"] == "synthetic":
        if protocol["availability"] != "synthetic" or cohort["split_path"] is not None:
            raise ValueError("synthetic protocols must declare manifest-only inputs")
        check_ids(cohort.get("case_ids"))
        if cohort["split_sha256"] != json_digest(cohort["case_ids"]):
            raise ValueError("synthetic cohort hash mismatch")
    elif not isinstance(cohort["split_path"], str) or not re.fullmatch(r"[a-f0-9]{64}", cohort["split_sha256"] or ""):
        raise ValueError("real protocols require an explicit split path and SHA256")
    return protocol


def load_protocol(path):
    return validate(json.loads(Path(path).read_text(encoding="utf-8")))


def requested_ids(protocol):
    """Verify the full requested split before exclusions. Never silently deduplicate or truncate."""
    if protocol["availability"] == "unavailable":
        raise ValueError(protocol["unavailable_reason"])
    cohort = protocol["cohort"]
    if protocol["adapter"] == "synthetic":
        ids = cohort["case_ids"]
    else:
        path = Path(cohort["split_path"])
        path = path if path.is_absolute() else ROOT / path
        if digest(path) != cohort["split_sha256"]:
            raise ValueError("split SHA256 mismatch")
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    check_ids(ids)
    if len(ids) != cohort["expected_count"]:
        raise ValueError("requested cohort count mismatch")
    if not set(cohort["exclusions"]) <= set(ids):
        raise ValueError("exclusions contain missing case IDs")
    return ids


@dataclass
class Pose:
    pose_id: str
    affinity: float
    rmsd: Optional[float] = None
    pb_valid: Optional[bool] = None
    error: Optional[str] = None
    pb_error: Optional[str] = None

    def __post_init__(self):
        if not self.pose_id or not math.isfinite(self.affinity):
            raise ValueError("pose identity and finite affinity required")
        if self.rmsd is not None and (not math.isfinite(self.rmsd) or self.rmsd < 0):
            raise ValueError("RMSD must be finite and nonnegative")
        if self.pb_valid is not None and type(self.pb_valid) is not bool:
            raise ValueError("PB validity must be true, false or unknown")


@dataclass
class Outcome:
    case_id: str
    seed: int
    run_label: str
    preparation: str
    status: str = "pending"
    stage: str = "inputs"
    error: Optional[str] = None
    stages: dict = field(default_factory=dict)
    input_digests: dict = field(default_factory=dict)
    artifact_digests: dict = field(default_factory=dict)
    produced_pose_count: int = 0
    poses: list = field(default_factory=list)

    @property
    def evaluated_pose_count(self):
        return sum(p.rmsd is not None for p in self.poses)

    def record(self, top_k):
        result = asdict(self)
        result["evaluated_pose_count"] = self.evaluated_pose_count
        result["metrics"] = pose_metrics(self, top_k)
        return result


def pose_metrics(outcome, top_k=5):
    """Reference-oracle sampling and affinity-ranked deployment metrics remain separate."""
    poses = sorted(outcome.poses, key=lambda p: (p.affinity, p.pose_id))
    if len({p.pose_id for p in poses}) != len(poses):
        raise ValueError("duplicate pose identity")
    evaluated = [p.rmsd for p in poses if p.rmsd is not None]
    ok = outcome.status == "ok"

    def success(selected, pb=False):
        return ok and any(p.rmsd is not None and p.rmsd <= 2 and (not pb or p.pb_valid is True) for p in selected)

    return {
        "oracle_min_rmsd": min(evaluated) if evaluated else None,
        "oracle_success_le2": success(poses),
        "top1_success_le2": success(poses[:1]),
        "topk_success_le2": success(poses[:top_k]),
        "top1_pb_valid_success_le2": success(poses[:1], pb=True),
        "topk_pb_valid_success_le2": success(poses[:top_k], pb=True),
        "pb_known_poses": sum(p.pb_valid is not None for p in poses),
        "pb_total_poses": len(poses),
        "top1_pose_id": poses[0].pose_id if poses else None,
    }


def summarize(outcomes, top_k=5):
    """Every eligible requested case/seed stays in success denominators, including failures."""
    keys = [(o.case_id, o.seed) for o in outcomes]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate case/seed outcome")
    if any(o.status not in ("ok", "failed", "excluded") for o in outcomes):
        raise ValueError("cannot summarize unfinished outcomes")
    eligible = [o for o in outcomes if o.status != "excluded"]
    metrics = [pose_metrics(o, top_k) for o in eligible]
    medians = [m["oracle_min_rmsd"] for m in metrics if m["oracle_min_rmsd"] is not None]
    total = len(eligible)
    result = {
        "requested_case_seeds": len(outcomes),
        "excluded_case_seeds": len(outcomes) - total,
        "eligible_case_seeds": total,
        "evaluated_case_seeds": len(medians),
        "failed_case_seeds": sum(o.status == "failed" for o in eligible),
        "oracle_median_rmsd_evaluated_only": statistics.median(medians) if medians else None,
        "top_k": top_k,
        "pb_known_poses": sum(m["pb_known_poses"] for m in metrics),
        "pb_total_poses": sum(m["pb_total_poses"] for m in metrics),
    }
    for name in (
        "oracle_success_le2",
        "top1_success_le2",
        "topk_success_le2",
        "top1_pb_valid_success_le2",
        "topk_pb_valid_success_le2",
    ):
        count = sum(m[name] for m in metrics)
        result[name] = {"count": count, "denominator": total, "percent": count / total * 100 if total else 0.0}
    return result
