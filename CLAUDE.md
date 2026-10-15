# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Research project on **unsupervised, tonality-agnostic harmonic boundary detection** in symbolic music (MusicXML). The core idea: build a hierarchical-clustering dendrogram over all notes and use the **merge height** of each pair of temporally adjacent notes as a resolution-independent boundary-strength score — no fixed window size, cluster count, or hand-tuned threshold.

The full proposal (motivation, research questions, prior work, evaluation design) is in `initial-proposal.md` — read it before making methodology decisions. (The project originally focused on persistent homology / `ripser` loops around cadences; that framing is obsolete. Ignore references to it.)

### Research questions (pursued in parallel)
1. Can harmonic boundaries be found from pitch/voice-leading data without fixed-resolution parameters?
2. Do discovered boundaries align with expert Roman-numeral annotations better than fixed-window / rule-based baselines?
3. Are boundaries stable across clustering resolution (merge height / persistence), even before segments can be labeled?
4. Does boundary quality hold on post-tonal / non-functional repertoire?

### Planned pipeline (not yet implemented)
1. **Parse** MusicXML with `music21` into per-note records.
2. **Featurize**, two parallel strategies:
   - Hand-crafted: pitch class, duration, onset, melodic interval & contour (same voice), metric position/beat strength, local note density, 12-dim co-occurring pitch-class vector at the note's onset (`chordify()`).
   - Learned: Magenta Polyphony RNN hidden states (512-dim, Bach chorale checkpoint).
   Keep every feature row traceable back to its note/onset.
3. **Cluster** hierarchically (scipy / scikit-learn); boundary score = adjacent-note merge height. Follow-up option: ToMATo persistence-based clustering (GUDHI / `tomaster`).
4. **Evaluate**:
   - Tonal sanity check: Bach chorales (music21 corpus), Roman-numeral annotations for Riemenschneider 1–20. Clusters labeled by majority vote of ground-truth labels; compare on a common fine timeline (e.g. 16th-note grid), not per measure. Report accuracy across *every* dendrogram cut (curve) plus the cut at the largest consecutive merge-height jump. Baseline: Uehara (2024) neural HSMM (`github.com/yui-u/harmonic-analysis-chorales`) — tonal evaluation only.
   - Atonal: reduce segments to weighted 12-dim pitch-class vectors (weight = duration × metric strength × ablation sensitivity), DTW against the original note-level pitch-class sequence, and report compression ratio (segments/notes) alongside. Note the known circularity of ablation-based importance.

## Environment & commands

- Package/env management via `uv` (`uv.lock`, `.venv`). Python >=3.12.
- Install/sync: `uv sync`; add deps with `uv add <pkg>`.
- Developed as marimo notebooks (plain `.py` files, e.g. `data-prep.py`):
  - `uv run marimo edit <file>.py` / `uv run marimo run <file>.py`
  - Edit cells as `@app.cell` functions; don't hand-edit `__marimo__/session/*.json`.

## Data layout

- **Exploratory corpus:** [PDMX](https://zenodo.org/records/15571083) (250K+ public-domain MusicXML scores + `PDMX.csv` metadata). `data-prep.py` downloads `mxl.tar.gz` and `PDMX.csv` into `data/raw/pdmx/` and extracts to `data/raw/pdmx/mxl/` (skips MIDI/PDF/JSON formats).
- **Validation corpus:** Bach chorales from the `music21` built-in corpus (371 chorales) with Roman-numeral annotations for a 20-piece subset — accessed via music21, not downloaded by `data-prep.py`.
- `data/` is gitignored; datasets are fetched/prepared locally only.
