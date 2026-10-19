# CLAUDE.md

## Project overview

Research project on **unsupervised, tonality-agnostic harmonic boundary detection** in symbolic music (MusicXML). The core idea: build a hierarchical-clustering dendrogram over all notes and use the **merge height** of each pair of temporally adjacent notes as a resolution-independent boundary-strength score — no fixed window size, cluster count, or hand-tuned threshold.

The full proposal (motivation, research questions, prior work, evaluation design) is in `initial-proposal.md` — read it before making methodology decisions. (The project originally focused on persistent homology / `ripser` loops around cadences; that framing is obsolete. Ignore references to it.)

### Research questions
1. Can harmonic boundaries be found from pitch/voice-leading data without fixed-resolution parameters?
2. Do discovered boundaries align with expert Roman-numeral annotations better than fixed-window / rule-based baselines?
3. Are boundaries stable across clustering resolution (merge height / persistence), even before segments can be labeled?
4. Does boundary quality hold on post-tonal / non-functional repertoire?

### Proposed pipeline
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

- `data/` is gitignored; datasets are fetched/prepared locally only.
- **Validation corpus:** Bach chorales from the `music21` built-in corpus (371 chorales) with Roman-numeral annotations for a 20-piece subset — accessed via music21, not downloaded by `data-prep.py`.

### Exploratory corpus: PDMX

[PDMX](https://zenodo.org/records/15571083): 254,077 public-domain MusicXML scores from MuseScore plus `PDMX.csv` metadata (62 columns). `data-prep.py` downloads and extracts it (skips MIDI/PDF/JSON formats):

- `data/pdmx/PDMX.csv` — metadata (note: `data/pdmx/`, not `data/raw/pdmx/`).
- `data/pdmx/mxl/` — scores. The CSV `mxl` column holds paths like `./mxl/1/11/Qm….mxl`, relative to `data/pdmx/`; open with `DATA_DIR / mxl[2:]`.34

**Metadata columns** (all derived without reading the score):
- Identity: `title`, `song_name`, `subtitle`, `artist_name`, `composer_name`, `genres`, `tags`, `groups`.
- Size/structure: `n_tracks`, `tracks`, `complexity`, `n_notes`, `notes_per_bar`, `n_tokens`, `song_length` (+ `.seconds`, `.bars`, `.beats`), `n_lyrics`, `n_annotations`.
- Precomputed musical features: `pitch_class_entropy` (bits, 0–3.58 = log2(12)), `scale_consistency` (fraction of notes in the best-fit major scale, 0.58–1.0), `groove_consistency`.
- Popularity/quality: `rating`, `n_ratings`, `n_views`, `n_favorites`, `n_comments`, flags (`is_official`, `is_original`, `is_draft`, `has_paywall`, ...).
- Licensing: `license`, `license_conflict`.
- Subset flags: `subset:all`, `rated`, `deduplicated`, `rated_deduplicated`, `no_license_conflict`, `all_valid`.

**Missing values:** `composer_name` ~64%, `genres` ~67%, `tags` ~92%, `groups` ~97%, `subtitle` ~85%, `song_name`/`artist_name` ~6%, `publisher` 100%. Composer names are sometimes mojibake or contain credits ("Arranged by ...", "Transcribed by ..."), so name matching is lossy. Match on composer + artist, not composer alone.

**Subsets** (size / with `scale_consistency` < 0.85):
| subset | scores | low-sc |
|---|---|---|
| `all_valid` | 254,035 | 10,261 |
| `deduplicated` (D) | 102,635 | 8,479 |
| `rated` | 14,182 | 2,825 |
| `rated_deduplicated` | 13,187 | 2,578 |

Most scores are unrated (median rating 0). `all_valid` includes lots of junk; **say which subset is used** — the working choice is `deduplicated`.

**What `scale_consistency` / `pitch_class_entropy` show:**
- Heavily tonal corpus: median consistency 1.00, 25th percentile 0.96; low-consistency (<0.85) is ~4%.
- Entropy and consistency are largely redundant: a ceiling means high consistency forces entropy <= ~2.8 bits (7 notes).
- Low-consistency scores are mostly chromatic common-practice, modulating, or messy transcriptions, not post-tonal. Whole-piece consistency alone does not separate modulation from non-tonal music; 8-bar windowed consistency does (see EDA in `data-prep.py`).

**Post-tonal coverage is thin.** In the deduplicated pool (100-3,000 notes, <=3 tracks), the strict composer list (Schoenberg/Schönberg, Webern, Alban Berg, Messiaen, Boulez, Ligeti, Stockhausen, Xenakis, Penderecki, Elliott Carter, Milton Babbitt, John Cage, Takemitsu, Lutosławski, Berio, Crumb) matches only **5 real pieces** (Ligeti, Messiaen, Webern, Berg, Schönberg). Beware false matches: "Carter", "Cage", "Babbitt" (hymn writers), "Claude-Michel Schönberg", "Schonberger". Impressionist/early-20th-century is better covered (rated+dedup: Debussy 78, Ravel 48, Bartók 36, Satie 27, Scriabin 13, Prokofiev 8). RQ4 may need another source or a narrower scope.

## Current prototype and session notes

### Coding and collaboration conventions

- For live marimo work, put implementation code in visible notebook cells via marimo code mode; scratchpad execution is for inspection and experiments, not the durable implementation.
- Work through the plan in small, reviewable steps and keep the user updated at checkpoints. If an ambiguity needs a user decision, pause and give a concrete example; mention non-blocking methodological caveats at the end.
- While a marimo session is active, treat the live notebook as source of truth and do not edit its `.py` artifact directly. Validate the saved artifact with `uv run marimo check <notebook.py>` and an export/run smoke check.
- Add comments to larger blocks of code, summarizing business logic and assumptions made

### `baseline.py` prototype status

The current exploratory notebook implements hand-crafted features and hierarchical clustering on the first annotated music21 Bach chorale, Riemenschneider 1, **“Aus meines Herzens Grunde.”** The score is loaded with `analysis=False`, excluding its Roman-numeral annotation part. Its four instrumental parts contain 229 note events total: Soprano 46, Alto 61, Tenor 59, Bass 63.

- One feature row is made per pitched note; chord pitches would be split, ties remain split, and grace notes are omitted. Each row retains source part/event/pitch indices and the source music21 element.
- Features currently include pitch class (0–11, one-hot encoded for clustering), duration in quarter lengths, absolute score-wide onset in quarter lengths, three-category beat strength, and a 12-dimensional count vector of pitch classes sounding at the onset. Simultaneous doublings count separately; sounding intervals are treated as `[onset, end)`.
- Beat strength is based on note onset: beat 1 is `strong`, the last metrical beat is `weak`, and other beats are `medium`. The chorale begins with a one-quarter pickup; music21 labels its onset beat 3 in 3/4, and it is classified `weak` per the agreed rule.
- Melodic interval, contour, and local note density are deferred. The feature matrix uses one-hot pitch class and beat strength plus unscaled duration, absolute onset, and co-occurrence counts in its initial clustering.
- Clustering uses SciPy average linkage with cosine distance and builds the full dendrogram. For 229 note rows it yields 228 merges and merge heights for 228 consecutive score-wide note pairs. Simultaneous-onset pairs are kept; ties are ordered by literal lexical part then pitch (source indices break any remaining ties).
- The displayed `adjacent_merge_heights` dataframe has one row per consecutive note pair and reports each pair’s first-common-cluster merge height; it is not a final segmentation or an evaluated boundary score.
- Three exploratory feature-scaling variants are present: unscaled numeric values; column-wise min–max scaling of all numeric columns; and min–max duration/onset with each co-occurrence count vector normalized to sum to one. No variant has been selected. Spearman rank correlations of adjacent-pair merge heights versus the unscaled variant were approximately 0.384 and 0.388 for the two scaled variants (the scaled variants correlated 0.757 with each other), indicating substantial sensitivity to feature scaling.
- Inline implementation checks passed for source traceability, onset count vectors, pickup handling, dendrogram dimensions/monotonicity, and scaling outputs. `marimo check baseline.py` and HTML export subsequently passed after the user copied the notebook into a fresh artifact. No Roman-numeral evaluation or baseline comparison has been performed.

### Deferred testing work

A pytest suite was discussed but not created. `pytest` is not currently configured as a project dependency, and a direct `import baseline` does not expose functions defined inside marimo cells as module-level attributes. The user chose to handle any small notebook restructuring needed before tests import those functions; do not extract notebook logic into a separate package unless asked.
