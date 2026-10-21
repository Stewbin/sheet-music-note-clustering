# Harmonic Boundary Detection via Hierarchical Clustering

This research project explores whether harmonic boundaries in symbolic music can be found from note and pitch-class information, without requiring chord labels, a fixed analysis window, or a predefined number of clusters. The current prototype parses MusicXML scores, builds note-level features, and fits a hierarchical clustering tree. Merge heights between adjacent notes are used to study candidate boundary strength across clustering resolutions.

The project is exploratory: the proposal describes the intended research, while the notebooks contain work-in-progress analysis. The current notebook pipeline should not be read as a completed evaluation or as a validated final method.

## Getting started

The project uses Python 3.12 or newer and [uv](https://docs.astral.sh/uv/) to manage its environment and dependencies.

1. Install uv if it is not already installed.
2. From the repository directory, install the project dependencies:

   ```bash
   uv sync
   ```

3. Open a notebook in marimo's interactive editor:

   ```bash
   uv run marimo edit clustering.py
   ```

   This opens the notebook in your browser. Keep the terminal process running while you use it; press **Ctrl+C** in that terminal to stop the server.

To open a notebook in app/viewer mode rather than edit mode, use `marimo run`:

```bash
uv run marimo run clustering.py
```

The same commands work for the other notebooks, for example `uv run marimo edit data-prep.py`.

### If you know Jupyter, what is different?

[Marimo notebooks](marimo.io) are still interactive notebooks with code cells and visible outputs, but they are saved as ordinary Python files (the `.py` files in this repository), rather than as `.ipynb` JSON documents. You can use the browser editor much like JupyterLab to inspect and edit code and outputs.

The main difference is execution. Marimo tracks which cells use the values defined by other cells and reruns affected cells when their inputs change. This reactive execution helps avoid the hidden, stale kernel state that can happen when Jupyter cells are run out of order. As a result, a variable should generally be defined in one cell and then used by other cells; avoid relying on re-running a cell in a particular order to overwrite a variable.

`marimo edit` is the interactive editor. To check a notebook for issues from the command line, run:

```bash
uv run marimo check clustering.py # replace with notebook of choice
```

To serve a notebook as an app for viewing and interacting with its controls, without editing its code, run:

```bash
uv run marimo run clustering.py
```

### Data requirements and downloads

The notebooks use two different sources of music data:

- The Bach chorale corpus and its annotations are accessed through `music21`.
- The exploratory PDMX MusicXML collection and metadata are stored locally under `data/pdmx/` and are not included in Git.

**Be aware:** `data-prep.py` downloads the PDMX metadata and a large MusicXML archive from Zenodo when those files are missing, then extracts the archive. Its cells execute as the notebook runs, so opening or running this notebook on a fresh checkout can start a substantial download. Run it only when you are ready to obtain the dataset and have enough disk space and network access. The notebook does not fetch the other PDMX formats that this project does not use.

The `clustering.py` notebook expects the local PDMX metadata and score files to be present. Prepare them with `data-prep.py` first. Generated/local data is ignored by Git, so it will not be committed with your code.

## Repository guide

| File or directory | Purpose |
| --- | --- |
| `clustering.py` | Main exploratory marimo notebook. Samples scores from the local PDMX `deduplicated` subset, parses them with `music21`, creates traceable note-level features, fits connectivity-constrained hierarchical clusters, and displays dendrograms and adjacent-note merge heights. This is the most developed analysis notebook. |
| `data-prep.py` | Marimo notebook for fetching/extracting PDMX data and exploring its metadata. Includes exploratory checks of scale consistency, pitch-class entropy, and short-window profiles for post-tonal and low-scale-consistency samples. Opening it can trigger the PDMX download described above. |
| `baseline.py` | Minimal marimo notebook stub. It is not the current working clustering pipeline; use `clustering.py` for the exploratory analysis. |
| `initial-proposal.md` | Initial research proposal: motivation, research questions, proposed features, clustering idea, and evaluation plans. It includes ideas that are still proposed rather than implemented. |
| `Harmonic Boundary Detection via Hierarchical Clustering Final Report.md` | Draft report describing the project's framing, datasets, methodology, and open decisions. Results and several sections are still placeholders. |
| `Null Annotations and Persistence Weight Scale.md` | Working note on evaluating boundary placement against annotations, persistence-weighted scores, null experiments, and baselines. |
| `figures/` | Tracked exploratory dendrogram and merge-height figures. These are example/earlier outputs, not a complete results set. |
| `tests/test_clustering.py` | Prototype validation checks for notebook-produced clustering outputs. It currently relies on objects from a live notebook session and is not a self-contained pytest suite. |
| `pyproject.toml` | Package metadata, Python requirement, and runtime/development dependencies. |
| `uv.lock` | Locked dependency versions used by `uv sync` for a reproducible environment. |
| `.python-version` | Python version selected for this project by version managers and uv. |
| `.gitignore` | Excludes local datasets, generated marimo session files, Python caches, and environment/build artifacts from Git. |
| `CLAUDE.md` | Detailed project context and contributor/agent guidance, including data notes and current prototype status. |

### Local and generated files

`data/` is deliberately ignored by Git. On a machine where the datasets have been prepared, it may contain PDMX files, downloaded Bach annotation resources, or generated intermediate data. Those files are local inputs/outputs, not source files required to understand the repository. The `__marimo__/session/` files are editor/session state and can also be ignored; the notebook source of record is the corresponding `.py` file.

## Research method at a glance

The notebook workflow represents score notes as feature rows, retaining source-part and note references so results can be traced back to the score. It fits a full hierarchical tree with a temporal adjacency constraint. Dendrogram merge heights for neighboring note rows are then inspected as candidate boundary-strength signals. The current features and scaling choices remain exploratory; no feature configuration has been selected as final, and the notebook does not yet constitute a complete Roman-numeral or baseline evaluation.

For the motivation and planned evaluation, see [`initial-proposal.md`](initial-proposal.md) and the draft [`Harmonic Boundary Detection via Hierarchical Clustering Final Report.md`](Harmonic%20Boundary%20Detection%20via%20Hierarchical%20Clustering%20Final%20Report.md).
