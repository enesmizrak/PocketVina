"""Dataset paths and existing molecular tools, imported only for real evaluation."""

import csv
import importlib.metadata
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from benchmark.protocol import Pose, digest


def _obabel():
    from pocketvina.ligands import find_obabel

    return find_obabel()


@dataclass(frozen=True)
class CaseFiles:
    protein: Path
    reference: Path
    smiles: Path

    def paths(self, recipe):
        paths = {"protein": self.protein, "reference": self.reference}
        if recipe != "reference-sdf-topology-historical":
            paths["smiles"] = self.smiles
        return paths


class DatasetAdapter:
    """Processed dataset layout used by the historical scripts; input files are read-only."""

    ligand_suffix = "_ligand.sdf"

    def locate(self, root, case_id):
        directory = Path(root) / case_id
        return CaseFiles(
            directory / f"{case_id}_protein_processed.pdb",
            directory / f"{case_id}{self.ligand_suffix}",
            directory / f"{case_id}_ligand.smiles",
        )


class PDBBindAdapter(DatasetAdapter):
    pass


class DockGenAdapter(DatasetAdapter):
    pass


class AstexAdapter(DatasetAdapter):
    ligand_suffix = "_ligands.sdf"


class PoseBustersAdapter(DatasetAdapter):
    ligand_suffix = "_ligands.sdf"


DATASETS = {
    "pdbbind": PDBBindAdapter,
    "dockgen": DockGenAdapter,
    "astex": AstexAdapter,
    "posebuster": PoseBustersAdapter,
}


def _command_version(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        return {"returncode": result.returncode, "output": (result.stdout + result.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"unavailable": str(error)}


def environment_identity():
    """Observe actual installed tools and engine/kernel bytes; never assume a tool version."""
    from pocketvina import __version__, engine, p2rank

    versions = {"pocketvina": __version__, "python": sys.version, "platform": platform.platform()}
    for package in ("rdkit", "openbabel-wheel", "networkx", "spyrmsd", "posebusters", "numpy", "meeko"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    exe = Path(engine.engine_path())
    kernel_root = Path(engine.kernel_source_dir())
    kernel_files = sorted(
        p for directory in engine.KERNEL_SOURCE_DIRS for p in (kernel_root / directory).rglob("*") if p.is_file()
    )
    if not exe.is_file() or not kernel_files:
        raise ValueError("real evaluation requires an existing engine and kernel sources for baseline identity")
    # A single tree identity, embedded in the report; no separate source manifest or archive.
    from benchmark.protocol import json_digest

    install = p2rank.find_p2rank()
    if install is None:
        raise ValueError("install P2Rank before collecting a real baseline identity (pocketvina setup p2rank)")
    p2rank_files = {}
    if install:
        home = Path(install.home)
        for path in sorted(home.rglob("*")):
            if path.is_file() and (path.suffix in (".jar", ".model") or path.name == "prank"):
                p2rank_files[str(path.relative_to(home))] = digest(path)
    java_home = os.environ.get("JAVA_HOME")
    java = str(Path(java_home) / "bin/java") if java_home else "java"
    return {
        "versions": versions,
        "engine_path": str(exe),
        "engine_sha256": digest(exe),
        "kernel_root": str(kernel_root),
        "kernel_sha256": json_digest({str(p.relative_to(kernel_root)): digest(p) for p in kernel_files}),
        "p2rank_home": install.home if install else None,
        "p2rank_tools_sha256": json_digest(p2rank_files) if install else None,
        "obabel": _command_version([_obabel(), "-V"]),
        "java": _command_version([java, "-version"]),
        "gpu": engine._gpu_identity(os.environ.get("CUDA_VISIBLE_DEVICES")),
    }


def _historical_preparation_mol(path, work):
    """The old scripts' sanitization, no-kekulization, OpenBabel and soft-load fallbacks, in order."""
    from rdkit import Chem

    def read(source):
        mol = next(iter(Chem.SDMolSupplier(str(source), removeHs=False, sanitize=False)), None)
        if mol is None:
            raise ValueError(f"Unable to load molecule from {source}")
        return mol

    mol = read(path)
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
        return mol, "normal"
    except Exception:
        try:
            mol = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol, kekulize=False), removeHs=False, sanitize=False)
            Chem.SanitizeMol(mol, sanitizeOps=(Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE))
            return mol, "without-kekulization"
        except Exception:
            try:
                # Historical repair commands, working on a copy so dataset inputs are never overwritten.
                copied = work / "reference.sdf"
                shutil.copyfile(path, copied)
                temporary, fixed = work / "reference_temp.mol", work / "reference_fixed.sdf"
                for source, target, fmt in ((copied, temporary, "-omol"), (temporary, fixed, "-osdf")):
                    subprocess.run(
                        [_obabel(), str(source), fmt, "-O", str(target), "--gen3d", "--noStereo"],
                        check=True,
                        capture_output=True,
                    )
                temporary.unlink()
                mol = read(fixed)
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
                return mol, "openbabel-repair"
            except Exception:
                mol = Chem.AddHs(Chem.RemoveHs(read(path), sanitize=False))
                code = Chem.SanitizeMol(mol, catchErrors=True)
                return mol, f"soft-load-sanitize-code-{int(code)}"


def prepare_case(files, case_id, recipe, seed, work):
    """Choose the recipe before creating a batch; return effective-preparation provenance."""
    target = work / "batch_1" / f"{case_id}_cleaned.pdbqt"
    target.parent.mkdir()
    if recipe == "reference-sdf-topology-historical":
        from benchmark.utils.utils import TopologyBuilder

        mol, path = _historical_preparation_mol(files.reference, work)
        topology = TopologyBuilder(mol)
        topology.build_molecular_graph()
        topology.write_pdbqt_file(str(target))
        details = {"recipe": recipe, "fallback": path, "embedding_seed": None, "receptor_preparation": "standard"}
    else:
        from pocketvina.ligands import prepare_ligands, read_smiles_file

        rows = read_smiles_file(files.smiles)
        if len(rows) != 1 or rows[0][0] != case_id:
            raise ValueError(f"SMILES table must have exactly one row with molecule_name={case_id}")
        method = "legacy" if recipe == "legacy-smiles" else "standard"
        record = prepare_ligands(rows, method=method, seed=seed)[0]
        if record.status != "ok" or not record.pdbqt:
            raise ValueError(f"ligand preparation: {record.status}: {record.error}")
        target.write_text(record.pdbqt, encoding="utf-8")
        details = {
            "recipe": recipe,
            "method": method,
            "smiles_input": record.smiles_input,
            "smiles_docked": record.smiles_docked,
            "embedding_seed": seed if method == "standard" else None,
            "receptor_preparation": method,
        }
    if not target.is_file() or not target.stat().st_size:
        raise ValueError("preparation produced no engine input")
    return dict(details, prepared_input_sha256=digest(target), engine_seed=seed)


class DockingOutputs(list):
    """Retain evaluable partial outputs while marking an incomplete docking case as failed."""

    def __init__(self, paths, errors):
        super().__init__(paths)
        self.errors = errors


def dock_case(files, protocol, seed, work):
    """Dock a fresh explicitly prepared batch using existing PocketVina code."""
    from pocketvina import P2RankConfig, PocketConfig, PocketVinaConfig, PocketVinaGPU

    proteins = work / "proteins.txt"
    proteins.write_text(str(files.protein.resolve()) + "\n", encoding="utf-8")
    # Historical topology batches deliberately bypass SMILES preparation; other recipes were also prepared above.
    smiles = work / "ligands.txt"
    smiles.write_text("molecule_name,smiles\n", encoding="utf-8")
    settings = protocol["settings"]
    box = settings["box_length"]
    config = PocketVinaConfig(
        protein_paths=str(proteins),
        smiles_file=str(smiles),
        output_dir=str(work),
        output_format="files",
        seed=seed,
        ligand_prep="legacy" if protocol["preparation"] == "legacy-smiles" else "standard",
        top_pockets=settings["top_pockets"],
        batch_size=10,
        p2rank_config=P2RankConfig(
            threads=settings["p2rank_threads"],
            model=settings["p2rank_model"],
            visualizations=False,
            vis_copy_proteins=False,
            output_dir=str(work),
        ),
        pocket_config=PocketConfig(size_x=box, size_y=box, size_z=box, thread=settings["vina_threads"]),
    )
    runner = PocketVinaGPU(config)
    errors = []

    class Errors(logging.Handler):
        def emit(self, record):
            errors.append(record.getMessage())

    handler = Errors(level=logging.ERROR)
    runner.logger.addHandler(handler)
    try:
        runner.execute_docking()
    finally:
        runner.logger.removeHandler(handler)
    outputs = sorted(work.glob("batch_*_out/*.pdbqt"))
    if not outputs:
        raise ValueError("docking produced no poses (inspect retained engine/P2Rank logs)")
    # The files runner logs engine failures and continues. Check every requested pocket, not just existing poses.
    from pocketvina.pdbqt import engine_output_name

    pocket_csv = work / "proteins/pockets" / f"{files.protein.stem}.csv"
    receptor = work / "proteins/structures" / f"{files.protein.stem}.pdbqt"
    with pocket_csv.open(newline="") as stream:
        reader = csv.DictReader(stream)
        ranks = [int(row[next(k for k in row if k.strip().lower() == "rank")]) for row in reader]
    if settings["top_pockets"] is not None:
        ranks = ranks[: settings["top_pockets"]]
    for rank in ranks:
        for ligand in (work / "batch_1").glob("*.pdbqt"):
            expected = work / "batch_1_out" / engine_output_name(str(receptor), rank, ligand.name)
            if not expected.is_file():
                errors.append(f"missing requested pocket output: {expected.name}")
    return DockingOutputs(outputs, errors)


def load_rmsd_mol(path):
    """The historical RMSD loader; retain its hydrogen and sanitization conventions."""
    from rdkit import Chem

    mol = next(iter(Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)), None)
    if mol is None:
        raise ValueError(f"Could not read molecule from {path}")
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
    except (Chem.KekulizeException, Chem.AtomValenceException):
        pass
    try:
        mol = Chem.RemoveHs(mol, implicitOnly=False, updateExplicitCount=True, sanitize=True)
    except Exception:
        pass
    return mol


def symmetry_rmsd(reference, candidate):
    """Same symmetry wrapper as the historical scripts, with molecular identity checked first."""
    import networkx as nx
    from spyrmsd.molecule import Molecule
    from spyrmsd.rmsd import rmsdwrapper

    ref = Molecule.from_rdkit(load_rmsd_mol(reference))
    cand = Molecule.from_rdkit(load_rmsd_mol(candidate))
    ref.strip()
    cand.strip()

    def graph(mol):
        result = nx.from_numpy_array(mol.adjacency_matrix)
        nx.set_node_attributes(result, dict(enumerate(map(int, mol.atomicnums))), "element")
        return result

    if not nx.is_isomorphic(
        graph(ref), graph(cand), node_match=nx.algorithms.isomorphism.categorical_node_match("element", None)
    ):
        raise ValueError("candidate/reference heavy-atom molecular identity mismatch")
    value = float(rmsdwrapper(ref, cand, symmetry=True, minimize=False)[0])
    if not math.isfinite(value) or value < 0:
        raise ValueError("RMSD tool returned a nonfinite or negative value")
    return value


def pb_validity(sdf, protein):
    """Use the runtime's dock physical checks. Missing or failed checks stay unknown."""
    from pocketvina import bust

    _, config = bust._config("dock")
    names = bust.check_names("dock")
    molecule = bust._load_protein(Path(protein).read_text(), config)
    if molecule is None:
        raise ValueError("PoseBusters protein load failed")
    result = bust._check_pose(bust._new_posebusters(config), names, ("sdf", Path(sdf).read_bytes()), molecule, None)
    values = result.get("values", [])
    if len(values) != len(names) or any(v is None for v in values):
        raise ValueError("PoseBusters checks are incomplete")
    return all(values)


def evaluate_outputs(files, outputs, work, outcome, use_pb):
    """Convert each MODEL separately, preserving the affinity/coordinates row identity on failure."""
    from pocketvina.pdbqt import check_modes

    directory = work / "evaluation"
    directory.mkdir()
    for output in outputs:
        raw = output.read_bytes()
        modes = check_modes(raw)
        outcome.produced_pose_count += len(modes)
        for mode in modes:
            pose_id = f"{output.stem}-mode{mode.mode:03d}"
            pose = Pose(pose_id, mode.affinity)
            outcome.poses.append(pose)
            pdbqt, sdf = directory / f"{pose_id}.pdbqt", directory / f"{pose_id}.sdf"
            pdbqt.write_bytes(raw[mode.raw_offset : mode.raw_offset + mode.raw_length])
            try:
                subprocess.run([_obabel(), str(pdbqt), "-osdf", "-O", str(sdf)], check=True, capture_output=True)
                pose.rmsd = symmetry_rmsd(files.reference, sdf)
            except Exception as error:
                pose.error = f"conversion/RMSD: {type(error).__name__}: {error}"
            if use_pb and sdf.is_file():
                try:
                    pose.pb_valid = pb_validity(sdf, files.protein)
                except Exception as error:
                    pose.pb_error = f"PoseBusters: {type(error).__name__}: {error}"
    if not outcome.poses:
        raise ValueError("no evaluable pose records")


def artifact_digests(work):
    return {str(p.relative_to(work)): digest(p) for p in sorted(work.rglob("*")) if p.is_file()}


def write_pose_csv(path, outcomes):
    """Pose-level rows keep the old RMSD columns plus score, seed and explicit evaluation status."""
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["Candidate_SDF", "Reference_SDF", "pdb_id", "RMSD", "affinity", "seed", "pb_valid", "error", "pb_error"]
        )
        for outcome in outcomes:
            for pose in outcome.poses:
                writer.writerow(
                    [
                        f"seed-{outcome.seed}/{outcome.case_id}/evaluation/{pose.pose_id}.sdf",
                        outcome.input_digests.get("reference", {}).get("path", ""),
                        outcome.case_id,
                        pose.rmsd,
                        pose.affinity,
                        outcome.seed,
                        pose.pb_valid,
                        pose.error,
                        pose.pb_error,
                    ]
                )
