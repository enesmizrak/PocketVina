# Reproducible benchmark protocols

Run from the repository root. These tools are repository-only: wheels and source distributions deliberately
exclude `benchmark/`. Protocol validation, accounting, help and dry-run use only the Python standard library.
Real preparation/evaluation uses PocketVina's existing tools plus the benchmark-only requirements:

```bash
python -m pip install -r benchmark/requirements.txt
python -m benchmark.run --help
python -m benchmark.run --protocol benchmark/protocols/smoke.json --dry-run
```

The smoke protocol declares three **simulated case IDs**, two seeds and six planned jobs. It creates no files,
requires no RDKit, Java, P2Rank or GPU, and refuses real docking. It makes no accuracy claim.

## Cohorts and historical protocols

Bundled split bytes, SHA256 and unique nonblank counts are fixed in the manifests. Blank lines are ignored;
duplicate IDs, unsafe IDs, changed hashes, count mismatches and exclusions absent from the split are errors.
Exclusions require reasons and are applied only after verifying the requested split. Missing input files remain
failed requested cases. No original split list has been edited or missing case ID inferred.

| Protocol | Requested cases | Status |
| --- | ---: | --- |
| `pdbbind-bundled357` | 357 | Bundled timesplit subset |
| `pdbbind-unseen-bundled144` | 144 | Bundled unseen split |
| `posebusters-bundled427` | 427 | Bundled PoseBusters subset |
| `astex-bundled85` | 85 | Bundled Astex split |
| `pdbbind-paper-full-unavailable` | 363 | Authoritative full split/exclusions not reconciled |
| `posebusters-paper-full-unavailable` | 428 | Authoritative full split/exclusions not reconciled |
| `dockgen-external-template` | unknown | External split, hash and count must be supplied |

All benchmark data and results of the paper, including TargetDock-AI, are on
[Zenodo](https://zenodo.org/records/15733460). The
[paper](https://arxiv.org/html/2506.20043v1) uses full requested cohorts for success percentages and evaluated
cases for median RMSD, and distinguishes sampling from scoring-selected poses. Historical script accounting and
preparation drift does not establish that published numbers were wrong. A bundled subset run must not be called
a full-paper reproduction. To enable an unavailable protocol, create a separately named definition with the
authoritative split, its byte hash and requested count, documented exclusions and availability `external`.

## Preparation, randomness and paths

Version 1 manifests specify the adapter, dataset root, split path/hash/count, exclusions, explicit seeds,
preparation recipe, box size, GPU search threads, P2Rank threads/model, pocket selection, label, ranking k and
whether to run PoseBusters. Relative split paths resolve against the repository root. Dataset roots resolve from
the invocation directory, so absolute paths are recommended. A null dataset root requires `--dataset-root` for
real execution. Adapters read these processed paths per ID, without modifying dataset files:

- All datasets: `<root>/<id>/<id>_protein_processed.pdb`.
- PDBbind and DockGen references: `<root>/<id>/<id>_ligand.sdf`.
- Astex and PoseBusters references: `<root>/<id>/<id>_ligands.sdf`.
- SMILES recipes: `<root>/<id>/<id>_ligand.smiles`, a CSV table with `molecule_name,smiles` header and exactly
  one row whose name equals the requested ID. A `.smiles` suffix does not change that existing CSV contract.

Three explicit recipes reuse existing chemistry, without new protonation or minimization defaults:

| Recipe | Effective ligand preparation | Receptor preparation |
| --- | --- | --- |
| `reference-sdf-topology-historical` | Reference SDF coordinates through the attributed, unchanged `TopologyBuilder` | Current standard |
| `legacy-smiles` | Existing legacy whole-SMILES, unseeded ETKDG/OpenBabel preparation | Existing legacy |
| `standard-smiles-seeded` | Existing largest-fragment, seeded ETKDGv3/OpenBabel preparation | Current standard |

The topology recipe preserves the old sanitization, no-kekulization, OpenBabel repair and soft-load fallback
order; repair scratch files live in the output directory. The selected fallback is recorded. It is explicitly
reference-coordinate preparation, so it is scientifically different from either SMILES recipe. Merely setting
`ligand_prep="legacy"` on a pre-created topology batch never selected legacy SMILES preparation in the old scripts.
The new runner selects and completes the recipe first, records the prepared PDBQT hash, then deliberately hands
that batch to the existing files runner. Every case/seed starts in a new directory; existing reports are refused.

Every engine call gets its declared seed. Standard embedding also gets that seed; legacy embedding remains
unseeded by its existing definition. Run labels and historical `--run_number` change artifact labels only and
never imply independent randomness. Two seeds need not produce different poses. Versions alone do not promise
bitwise reproducibility across platforms or tool changes.

Historical entry points are thin wrappers around the shared runner, including direct script invocation:

```bash
python -m benchmark.astex.main --dataset-root /data/astex --results_dir results/astex-baseline
python benchmark/pdbbind/main.py --dry-run --run_number 2 --vina_threads 8000
```

The old `--box_length`, `--vina_threads`, `--p2rank_threads`, `--p2rank_model` and `--results_dir` spellings remain.
The protocol printed by dry-run and embedded in reports contains all effective overrides. DockGen requires an
explicit `--protocol` with a verified external split; there is no author-local default path.

## Reports and accuracy interpretation

`report.json` is the authoritative ledger: one outcome per requested case/seed, including exclusions and failures,
with stages, errors, effective preparation, seeds, counts, input/artifact hashes and separate pose metrics.
`outcomes.csv` has the same case/seed grain and `poses.csv` has pose-level RMSD and affinity rows. All three files
are written with atomic replacement, including header-only pose output and a summary when every case fails.
`definition.json` saves the protocol and pre-run environment identity before case execution. Raw docking outputs
and conversion artifacts are retained. Historical raw PDBQT/JSON/CSV files remain readable by their existing
readers; the new runner does not overwrite or append old CSV summary rows.

Success denominators include **all eligible requested case/seed pairs**, after declared exclusions. Each seed also
has its own summary. Pooled summaries count case/seed pairs; they do not combine seeds to choose a better pose.
Missing inputs, preparation failures, missing requested pocket outputs, logged engine errors, conversion errors
and evaluation errors mark the case failed. A partially evaluated failed case contributes no success; its available
RMSDs remain inspectable and enter the explicitly labeled evaluated-only median. Counts refer to parsed produced
poses and successfully RMSD-evaluated poses, respectively.

- `oracle_success_le2` uses the minimum reference RMSD over sampled poses: a sampling metric.
- `top1_success_le2` and `topk_success_le2` rank by ascending affinity across all produced poses. Ties use lexical
  `pose_id` order, independent of input order. Reference RMSD never chooses a deployable top pose.
- `top1_pb_valid_success_le2` and `topk_pb_valid_success_le2` additionally require observed PoseBusters validity.
  The `dock` physical checks are reused. Missing, failed or incomplete checks are unknown, never valid.
  `pb_known_poses` / `pb_total_poses` reports coverage, including when PB is disabled. A PB error when requested is
  retained separately on the pose and marks the PB stage incomplete; observed RMSD metrics remain usable.
  No PB-valid success is inferred from a missing result.
- `oracle_median_rmsd_evaluated_only` is the median of each evaluated case/seed's minimum RMSD. It is null when
  nothing evaluates; it is never a full-cohort accuracy estimate by itself.

Conversion splits each engine MODEL before OpenBabel SDF conversion, retaining its own affinity and stable pose ID.
Reference and candidate heavy-atom element/bond graphs must be isomorphic before comparing coordinates. RMSD reuses
`spyrmsd.rmsdwrapper(symmetry=True, minimize=False)`, `Molecule.strip()` and the historical hydrogen-removal loader:
symmetry corrected, no coordinate alignment, no added bond-order reconstruction. Tool versions and these conventions
are embedded in reports; chemistry or identity errors are not silently repaired during RMSD evaluation.

## Frozen pilot and CPU evidence

The inexpensive first real pilot is **`astex-bundled85.json` unchanged**, seeds **42 and 43**, historical topology
preparation, 30 Angstrom boxes, 8000 engine threads, 8 P2Rank threads, `alphafold` model and all predicted pockets.
Use the available processed dataset and a new ignored results directory:

```bash
pocketvina setup p2rank
python -m benchmark.run --protocol benchmark/protocols/astex-bundled85.json \
  --dataset-root /data/astex --results-dir results/astex-baseline-v1 --dry-run
python -m benchmark.run --protocol benchmark/protocols/astex-bundled85.json \
  --dataset-root /data/astex --results-dir results/astex-baseline-v1
```

Install P2Rank before a real run so the initial environment identity describes the actual tools, without an
implicit download during identity collection. Reports record observed Python/package/Java/OpenBabel versions,
P2Rank launcher/JAR/model identity, engine bytes, kernel source identity and available GPU/driver identity before
docking, plus per-case inputs and prepared-input/artifact hashes. Missing optional packages are recorded as null,
not guessed versions. Preserve this baseline report and the dataset before accepting later numerical changes;
compare the same requested cohort, exclusions, recipes, seeds, settings and input/tool identities.

The CPU foundation tests validate accounting/orchestration, recipe/seed forwarding through a fake engine, historical
symmetry convention and fixture conversion. CI runs the dependency-light contracts and smoke dry-run separately;
scientific integration runs in the normal CPU suite. Full datasets and real GPU runs remain opt-in. At implementation
time the macOS environment has no installed P2Rank or observed NVIDIA GPU identity, and no real dataset/GPU pilot
has been run. No empirical accuracy baseline is claimed. CPU completion does **not** establish plan 014 readiness;
a real GPU baseline is required before a dependent numerical search change is accepted.
