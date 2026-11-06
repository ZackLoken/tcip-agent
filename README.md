# TCIP Agent

An agentic ML/CV platform for automated phenotyping in tree-crop breeding programs, entering alpha
testing. An agent (Claude Code, Codex or Antigravity, or any MCP client) does the work of a
computer-vision engineer: it designs the pipeline, trains and evaluates models, and delivers
per-plant results through the platform's MCP tools. A browser GUI is where the breeder labels
examples, confirms or corrects the model, confirms what each trait means, and receives the result.

Who it's for: plant breeders, not CV engineers. They know their crops and traits; they don't write
PyTorch or make CV decisions. The platform supplies what the agent cannot otherwise know (the
primitives and their interfaces, the trait definitions, the scientific rails, the data in hand) and
leaves the method to the agent's own reasoning. There are two users and two surfaces: the agent's
is the tool, knowledge and code surface it reasons through, under the operating contract
(`packages/tcip-mcp/src/tcip_mcp/knowledge/operating-contract.md`); the breeder's is the GUI, the
only surface they work through.

Scope today: 2D imagery, RGB and N-channel (GeoTIFF, NPZ, grayscale, and multispectral captures
stored one file per band), in two capture shapes: per-plant photos from a phone, DSLR or ground rig,
and a single georeferenced orthomosaic covering many plants, read window by window rather than
loaded whole. Task heads are provided for detection, instance and semantic segmentation,
classification, ordinal and regression; the agent's own model is not limited to them. 3D point
clouds are not built.

Six crops are in scope: hazelnut, chestnut, currant, elderberry, persimmon and black locust. Their
traits are the controlled vocabulary in `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/crops.yml`.
The first pilot, used as the worked example below, is hazelnut catkin phenology.

## How a result is trusted

Every delivered phenotype (a count, a dimensional measurement, a phenology milestone) passes one
delivery gate (`gate` in `delivery.py`):

- The breeder confirms each trait's operationalization in the Setup tab; the agent proposes it with
  `propose_trait`, and only a confirmed revision lets a delivery proceed.
- A delivered number rests on an assessment: `assess_checkpoint` (or `assess_reserved_regions` for a
  mosaic, `calibrate_physical_scale` for a physical unit) fits the operating point on a selection's
  calibration side and checks it on the holdout side against the trait's confirmed criterion.
- A bucket that no passing assessment covers, for its producer, execution record and capture date,
  delivers only under an acknowledgment the breeder records in the Results tab for exactly that
  result. Some refusals take no acknowledgment at all: proposals, more than one producer, no
  positive state, a unit with no scale assessment.
- Every mutating door leaves exactly one audit line; experiments are immutable, a relaunch is a new
  run directory.

## Architecture

```text
┌──────────────────────────────┐         ┌──────────────────────────────┐
│  Agent harness               │  MCP    │  Python MCP server           │
│  Claude Code, Codex,         │ ──────▶ │  (packages/tcip-mcp)         │
│  Antigravity, any client     │  stdio  │  tools, pipelines, knowledge │
└──────────────▲───────────────┘         └──────────────┬───────────────┘
               │ spawned in the GUI's                    │ HTTP
               │ agent terminal                          ▼
┌──────────────┴───────────────────────────────────────────────────────┐
│  FastAPI backend + React GUI (packages/tcip-web)                      │
│  the breeder's tabs; the agent terminal; the tcip console dispatcher  │
└───────────────────────────────────────────────────────────────────────┘
```

Supporting libraries: `packages/tcip-annotation` (label documents, matching) and
`packages/tcip-store` (the storage seam: records and append-only logs in one SQLite database per
root, `<root>/.tcip/store.db`; imagery stays files). `tcip-store` depends on nothing here;
`tcip-annotation` depends only on it.

## What the agent works with

- MCP tools: run `python tools/list_tools.py` for the current list. They cover project setup and
  ingest, the subject registry and labels, auto-labeling proposals (SAM prompts and Grounding DINO
  text search ship as engines, with a registry for your own), splits, training and sweeps (Ray
  Tune), evaluation and failure renders, inference over photos or a whole mosaic, assessment,
  delivery, project memory (`report_friction`, `write_retrospective`, `load_project_memory`), and
  steering the GUI (`push_panel_event`, `focus_human_attention`, `capture_live_canvas`,
  `view_gui_state`).
- Domain knowledge: the documents under `packages/tcip-mcp/src/tcip_mcp/knowledge/`, served
  through `serve_domain_knowledge` and written by every terminal launch into the workspace as the
  skills its harness reads, beside the agent's operating contract in its instruction file.
- One build path: the agent writes its own `nn.Module` and `train(ctx)` loop over the platform's
  building blocks (heads, necks, losses, derivations) and a library's own detector, held to the
  model contract; see the `pipeline-design` and `toolkit-inventory` skills.

## The breeder's GUI

The GUI opens a workspace (a folder whose child directories are projects) and works on one project
at a time, in six tabs: Setup (the project record, capture invariances, trait confirmation, plant
mapping), Annotate (labeling, review of model proposals, completion marks and flags), Training
(runs, an embedded TensorBoard, run comparison), Inference (jobs over photos or a mosaic), Results
(deliveries and acknowledgments) and Meta (the project's friction reports and retrospectives). An
agent terminal beside the tabs launches Claude Code, Codex or Antigravity on the open project; a
launch resumes the row's last conversation when the backend's rule allows it and tells the breeder
why when it does not, and a switch of harness asks the running agent for a hand-off first.

## Setup

A fresh machine needs these before the first command:

- git, to clone the repository.
- Miniforge, the conda installer with conda-forge as its only channel; Node comes from the
  environment. Its installers are under
  `https://github.com/conda-forge/miniforge/releases/latest/download/`, named in Miniforge's
  README (`Miniforge3-Windows-x86_64.exe`, `Miniforge3-$(uname)-$(uname -m).sh`). A Miniconda
  install adds the Anaconda channels, whose terms only you can accept, and `conda env create`
  stops until you do.
- On Linux, the libGL and GLib libraries the opencv-python wheel links: on Debian and Ubuntu,
  `sudo apt-get update` then `sudo apt-get install -y libgl1 libglib2.0-0`, since a fresh
  machine holds no package index until the first update.
- For the agent terminal, the command line of each harness you will run, on `PATH` and signed in:
  `claude` (Claude Code), `codex` (Codex) or `agy` (Antigravity).
- Docker, for the container image alone.

Every command below runs from the repository root, in a shell conda is initialized for, alike in
bash and PowerShell except the one `export` line, whose PowerShell form sits beside it. Miniforge's
README says the Windows installer's defaults leave `conda` in the Miniforge Prompt alone until
`conda init` is run there ("conda/mamba usable in any terminals"), and that the Unix installer
prompts to initialize your shell ("Unix-like platforms").

```bash
# Python: creates the env and installs the four packages (editable).
conda env create -f environment.yml
conda activate tcip-agent

# Frontend: install, then build the bundle the backend serves (the built bundle is not tracked).
npm --prefix packages/tcip-web/frontend ci
npm --prefix packages/tcip-web/frontend run build
```

The torch build the environment installs, and the condition it holds under, is the one
`environment.yml`'s header states. On a machine with an NVIDIA driver, this line in the activated
environment replaces it with the CUDA 13.2 build, the one PyTorch's install page
(`https://pytorch.org/get-started/locally/`) lists first for torch 2.15.0:

```bash
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

On Apple silicon with macOS 14 or later the environment's torch is PyPI's build, built for Apple's
GPU, and the CUDA line is not run there.

The build writes the bundle inside the `tcip-web` package, so the environment's editable install
serves it at once, and records the source it was built from beside it: a backend started from a
checkout whose bundle was not built from the source beside it logs the command that rebuilds it
and shows a stale-bundle notice in the top bar. A wheel or the Docker image carries the bundle
built before it was made; a `v*` tag's workflow (`.github/workflows/release.yml`) builds the
bundle, then the wheel, and attaches the wheel to the tag's GitHub release.

An existing editable checkout gains the `tcip` console script only on the next
`pip install -e packages/tcip-web`.

`docker build -f packages/tcip-web/Dockerfile -t tcip .` from the repository root, then `docker run
--network host -e TCIP_WORKSPACE=<workspace> tcip`, builds and runs the platform as a container,
serving the GUI on the host's loopback. The image carries the CPU build of torch and no agent
harness, so the GUI's agent terminal does not run inside it. A published port (`-p`) cannot serve
the GUI, since its requests would arrive through a bridged address the backend's trust boundary
refuses.

## Running

```bash
# Backend plus the built GUI at http://127.0.0.1:8765; it refuses to start without the workspace.
export TCIP_WORKSPACE=<workspace>      # PowerShell: $env:TCIP_WORKSPACE = "<workspace>"
python -m tcip_web

# Frontend dev server, proxying /api and /ws to the backend, at http://127.0.0.1:5173
npm --prefix packages/tcip-web/frontend run dev

# The console: `tcip --help` lists every command with what it does.
tcip scan-dataset --project <project> --user <your name> --folder-path images
tcip dump-store <project> <out_dir>
```

`TCIP_WEB_PORT` overrides the backend's port, and a taken port falls back to one the OS assigns.
Every MCP tool is a `tcip` command of its name with hyphens, called with `--project` and the
`--user` it acts for; a few commands are for the operator alone (`archive-project`,
`import-project`, `write-project-record`, `dump-store`, `distill-learnings`).

An MCP client starts the server itself (see `.mcp.json`). A server acts on the project it was
started for (`python -m tcip_mcp --project <path>`), or, started for none, on the first project its
`initialize_project` creates; both need `TCIP_WORKSPACE`.

The tests and gates are `CLAUDE.md`'s Commands block; `python tools/smoke_terminal_e2e.py claude`
is one live run of a provider row, which spends model turns.

## From images to a delivered number

What an agent walks through on a first project, using the hazelnut catkin pilot as the worked
example. Each tool's full signature is in its description; this is the order, not the interface.

1. `initialize_project`: the project's display name, site and capture invariances, each asked of
   the breeder.
2. `ingest_images`: copies the photos into `images/<capture date>/` by the date each file states,
   undated ones under `undated/`; `state_capture_date` re-dates them.
3. `register_dataset` and `write_subject_registry`: the dataset's identity, and the subjects (object
   classes) and attributes the breeder names.
4. Labeling in the Annotate tab, by the breeder, with proposals from `propose_annotations` to accept
   or correct. An image with nothing to label is marked done there; an empty label document alone
   is never a negative.
5. `draw_splits` (or `plant_aware_group_splits`, grouping by plant): a leakage-free
   train/val/calibration/holdout selection.
6. `preflight_config`, then `launch_training` with the agent's own model source; `monitor_training`
   and `evaluate_model` follow it.
7. `propose_trait`, confirmed by the breeder in Setup; then `assess_checkpoint` on the selection.
8. `run_inference`: publishes predictions as a bucket under the assessment.
9. A delivery door: `deliver_per_image_counts`, `deliver_per_plant_csv` or
   `deliver_phenology_milestones` (the last two after `register_plant_registry` and
   `build_plant_mapping`) or `deliver_orthomosaic_plant_counts`. The `delivery` skill says which
   door answers which question.

## Roadmap

Built:

- Detection, instance and semantic segmentation, classification, ordinal and regression, through
  one training path with progressive unfreezing, SAHI slicing, a model contract every detection
  model meets before training, class-weighted and focal losses, augmentation each transform of
  which the project's capture invariances must admit, and Ray Tune sweeps.
- Native per-image label documents with provenance, model proposals reviewed on the canvas, and
  active-learning ranking of the next review batch (`prioritize_review_queue`).
- Whole-mosaic tiled inference with GDAL windowed reads, detections resolved to real-world
  coordinates and matched to a registered plant registry, or attributed to canopy segments.
- State-crossing milestones for any trait whose `crops.yml` entry states a positive state and
  milestone fractions, delivered per plant from per-date buckets.
- A physical-scale path for per-photo dimensional traits.
- The in-app agent terminal for three harnesses, resume of a row's last conversation, and the
  hand-off on a switch.

Not built yet:

- 3D point clouds (LiDAR, SfM): no loader or task type.
- Time-based primitives the toolkit does not provide: the `_julian` milestone traits `crops.yml`
  defines have no producer; a delivered milestone date is validated only through its inputs, not
  against a field-observed date; and a training sample or prediction bucket cannot span several
  captures of one plant. These are open work items, as primitives the agent would compose, not as a
  prescribed pipeline.
- A fully automated active-learning loop with no person in it.
- Plant identity without georeferencing (a QR code or tag on the plant).
- Harnesses beyond the three rows the agent terminal launches (any MCP client can still drive the
  server directly).
- Cloud or multi-user storage; the store is local SQLite and files.
- Hosted, multi-user deployment; the GUI binds to loopback as a single-operator desktop tool.

## Security and data egress

When the agent works, the breeding data it reads is sent to your model provider, as it would be if
you pasted it into a chat, and the platform does not bound or redact it. Read
[SECURITY.md](SECURITY.md) before using TCIP on commercially sensitive data: it lists every channel
that leaves the machine, the one phone-home you can disable, and the trust boundary the loopback
bind rests on.

Contributing: [CONTRIBUTING.md](CONTRIBUTING.md) says how to report a problem or ask for a feature,
and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) the standards contributors are held to. What an
adopter can build against today is [STABILITY.md](STABILITY.md)'s; [VERSIONING.md](VERSIONING.md)
explains what the version number does and doesn't promise.

## License

TCIP Agent is released under the [Apache License 2.0](LICENSE) (© 2026 Zack Loken). Commercial use
is permitted. Third-party attributions are in [NOTICE](NOTICE).
