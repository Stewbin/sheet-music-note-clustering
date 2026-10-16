import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import tarfile
    import urllib.request
    import pandas as pd
    from pathlib import Path
    from music21 import converter
    import numpy as np
    import re

    return Path, converter, mo, np, pd, tarfile, urllib


@app.cell
def _(Path):
    # Constants: PDMX dataset (Zenodo record 15571083)
    ZENODO_RECORD = "15571083"
    ZENODO_BASE = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"

    # Only the MusicXML archive + metadata CSV are fetched; music21 only
    # needs MXL, and the other formats (MIDI/PDF/MusicRender JSON) would add
    # ~12GB of downloads that this project doesn't use.
    MXL_URL = f"{ZENODO_BASE}/mxl.tar.gz/content"
    METADATA_CSV_URL = f"{ZENODO_BASE}/PDMX.csv/content"

    DATA_DIR = Path("data/pdmx")
    MXL_ARCHIVE = DATA_DIR / "mxl.tar.gz"
    METADATA_CSV = DATA_DIR / "PDMX.csv"
    return DATA_DIR, METADATA_CSV, METADATA_CSV_URL, MXL_ARCHIVE, MXL_URL


@app.cell
def _(DATA_DIR):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return


@app.cell
def _(MXL_ARCHIVE, MXL_URL, urllib):
    if not MXL_ARCHIVE.exists():
        print("Local data not found. Downloading from source.")
        urllib.request.urlretrieve(MXL_URL, MXL_ARCHIVE)
    return


@app.cell
def _(METADATA_CSV, METADATA_CSV_URL, urllib):
    if not METADATA_CSV.exists():
        print("Metadata not found. Downloading from source.")
        urllib.request.urlretrieve(METADATA_CSV_URL, METADATA_CSV)
    return


@app.cell
def _(DATA_DIR, MXL_ARCHIVE, tarfile):
    mxl_dir = DATA_DIR / "mxl"
    if not mxl_dir.exists():
        with tarfile.open(MXL_ARCHIVE) as tar:
            tar.extractall(DATA_DIR, filter="data")
    return


@app.cell
def top_composers_heading(mo):
    mo.md("""
    ## Top composers by genre
    """)
    return


@app.cell
def genre_df(METADATA_CSV, pd):
    genre_df = pd.read_csv(
        METADATA_CSV,
        usecols=["genres", "composer_name"],
    ).dropna()
    genre_df = genre_df[(genre_df["genres"] != "NA") & (genre_df["composer_name"] != "NA")]
    return (genre_df,)


@app.cell
def top_composers_by_genre(genre_df):
    TARGET_GENRES = ["classical", "jazz", "rock", "metal"]

    top_composers_by_genre = {genre: (genre_df[genre_df["genres"] == genre]["composer_name"].value_counts().head(5)) for genre in TARGET_GENRES}
    top_composers_by_genre
    return


@app.cell
def _(mo):
    mo.md("""
    ## Tonality proxies: scale consistency vs. pitch-class entropy
    - `pitch_class_entropy`: a high value means the pitch classes are spread evenly, which is a rough sign of less tonal music
    - `scale_consistency`: the share of notes that fit a single scale, so a low value suggests non-tonal or modulating music.
    """)
    return


@app.cell
def _(METADATA_CSV, pd):
    features_df = pd.read_csv(
        METADATA_CSV,
        usecols=["scale_consistency", "pitch_class_entropy"],
    ).dropna()
    return (features_df,)


@app.cell
def _(features_df):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        features_df["scale_consistency"],
        features_df["pitch_class_entropy"],
        s=2,
        alpha=0.05,
        linewidths=0,
    )
    ax.set_xlabel("scale_consistency")
    ax.set_ylabel("pitch_class_entropy")
    ax.set_title(f"PDMX scores (n={len(features_df):,})")
    fig
    return (plt,)


@app.cell
def _(pd):
    d = pd.read_csv("data/pdmx/PDMX.csv", low_memory=False)
    r = d[d["subset:deduplicated"]]
    txt = (r.composer_name.fillna("") + "|" + r.artist_name.fillna("") + "|" + r.title.fillna("")).str.lower()
    names = {
        "Schoenberg": "schoenberg|schönberg|schonberg",
        "Webern": "webern",
        "Berg": r"albanberg|\bberg,|berg\b",
        "Messiaen": "messiaen",
        "Debussy": "debussy",
        "Ravel": "ravel",
        "Stravinsky": "stravinsk",
        "Bartok": "bart[óo]k",
        "Scriabin": "scriabin|skrjabin|scriabine",
        "Ligeti": "ligeti",
        "Satie": "satie",
        "Prokofiev": "prokofiev",
        "Shostakovich": "shostakovich",
        "Ives": r"charles ives",
        "Cage": r"john cage",
        "Crumb": r"george crumb",
        "Hindemith": "hindemith",
        "Takemitsu": "takemitsu",
        "Xenakis": "xenakis",
        "Boulez": "boulez",
        "Stockhausen": "stockhausen",
        "Penderecki": "penderecki",
        "Ginastera": "ginastera",
        "Szymanowski": "szymanowski",
    }
    rows = []
    for k, p in names.items():
        m = txt.str.contains(p, regex=True)
        rows.append((k, m.sum(), round(r.scale_consistency[m].median(), 3) if m.sum() else None, (r.scale_consistency[m] < 0.9).sum()))
    print(pd.DataFrame(rows, columns=["name", "n", "median_sc", "n_sc<0.9"]).to_string(index=False))
    print(len(r))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Sliding-window scale consistency: post-tonal vs. low-consistency pieces
    """)
    return


@app.cell
def _(DATA_DIR, converter, np, pd):
    # Helper functions

    MAJOR_MASK = np.zeros(12, dtype=bool)
    MAJOR_MASK[[0, 2, 4, 5, 7, 9, 11]] = True

    def scale_consistency(pcs):
        """Largest fraction of notes inside any transposition of the major scale."""
        # Rotate the mask through all 12 keys and keep the fit.
        counts = np.bincount(pcs, minlength=12)
        return max(counts[np.roll(MAJOR_MASK, k)].sum() for k in range(12)) / counts.sum()

    def pc_entropy(pcs):
        """Shannon entropy (bits) of the pitch-class histogram; 0 = one pitch, ~3.58 =
        flat."""
        counts = np.bincount(pcs, minlength=12).astype(float)
        p = counts[counts > 0] / counts.sum()
        return -(p * np.log2(p)).sum()

    def window_profile(mxl_rel_path, width=8, hop=4, min_notes=8):
        """Per-window profile over the whole piece, plus the whole-piece scale consistency.

        Returns (DataFrame[start_bar, scale_consistency, entropy], whole_piece_sc).
        Windows with fewer than `min_notes` notes are kept with NaN values so the
        time axis has no gaps.
        """
        # Parse the MusicXML and flatten it to (measure number, pitch class) per sounding pitch.
        score = converter.parse(DATA_DIR / mxl_rel_path[2:])
        pairs = [(note.measureNumber or 0, pitch.pitchClass) for note in score.recurse().notes for pitch in note.pitches]
        bars, pcs = np.array(pairs).T
        # Slide the window from the first bar until its start passes the last bar,
        # so the (shorter) tail of the piece is covered too.
        rows = []
        for start in range(bars.min(), bars.max() + 1, hop):
            in_window = (bars >= start) & (bars < start + width)
            if in_window.sum() >= min_notes:
                rows.append((start, scale_consistency(pcs[in_window]), pc_entropy(pcs[in_window])))
            else:
                rows.append((start, np.nan, np.nan))
        profile = pd.DataFrame(rows, columns=["start_bar", "scale_consistency", "entropy"])
        # Also return the whole-piece value to compare against the CSV column.
        return profile, scale_consistency(pcs)

    return (window_profile,)


@app.cell
def _(METADATA_CSV, pd):
    # Load the full metadata and keep the candidate pool: the deduplicated subset,
    pool_df = pd.read_csv(METADATA_CSV, low_memory=False)
    pool_df = pool_df[pool_df["subset:deduplicated"] & pool_df["n_notes"].between(100, 3000) & (pool_df["n_tracks"] <= 3)].copy()

    # limited to short, few-track scores so music21 parsing stays fast.
    _credit = (pool_df["composer_name"].fillna("") + " | " + pool_df["artist_name"].fillna("")).str.lower()

    # One lowercase "credit" string per score (composer + artist) to match names against.
    # Explicit list of post-tonal composers (strict names, no bare surnames like "carter").
    # "schonberg(?!er)" skips the unrelated "Schonberger"; "milton babbitt" skips hymn writers named Babbitt.
    POST_TONAL_PATTERN = (
        r"schoenberg|schönberg|schonberg(?!er)|webern|alban berg|messiaen|boulez|ligeti"
        r"|stockhausen|xenakis|penderecki|elliott carter|milton babbitt|john cage|takemitsu"
        r"|lutosławski|lutoslawski|luciano berio|george crumb"
    )
    # Exclude the musical-theatre composer Claude-Michel Schönberg
    _is_post = _credit.str.contains(POST_TONAL_PATTERN, regex=True) & ~_credit.str.contains("claude-michel")
    pool_df["credit"] = _credit
    post_pool = pool_df[_is_post]
    # Show the matches so they can be checked by eye before sampling.
    post_pool[["title", "composer_name", "n_notes", "scale_consistency"]]
    return POST_TONAL_PATTERN, pool_df, post_pool


@app.cell
def _(POST_TONAL_PATTERN, pd, pool_df, post_pool, window_profile):
    # Comparison group: low-consistency scores that are not by post-tonal or other
    # known non-diatonic composers, so we test what the metric catches on its own.
    known_non_atonal = r"debussy|ravel|scriabin|bart[óo]k|stravinsk|prokofiev|satie|ives"
    low_sc_pool = pool_df[(pool_df["scale_consistency"] < 0.85) & ~pool_df["credit"].str.contains(POST_TONAL_PATTERN + "|" + known_non_atonal, regex=True)]
    # Sample 5 pieces from each group (fixed seed so the draw is reproducible).
    sample_post = post_pool.sample(5, random_state=1)
    sample_low = low_sc_pool.sample(5, random_state=1)

    # Run the sliding-window analysis over every piece. Keep each piece's full
    # profile (for plotting) and one summary row (CSV value vs. our recomputed
    # whole-piece value as a sanity check, plus window median / min / max and
    # median entropy, ignoring empty windows).
    window_profiles = {}
    _rows = []
    for _group, _sample in [("post-tonal", sample_post), ("low-sc", sample_low)]:
        for _, _r in _sample.iterrows():
            _prof, _whole = window_profile(_r["mxl"])
            _label = f"{str(_r['title'])[:38]} ({str(_r['composer_name'])[:20]})"
            window_profiles[_label] = (_group, _prof)
            _rows.append(
                dict(
                    group=_group,
                    title=str(_r["title"])[:38],
                    composer=str(_r["composer_name"])[:20],
                    csv_sc=round(_r["scale_consistency"], 3),
                    my_sc=round(_whole, 3),
                    win_sc_median=round(_prof["scale_consistency"].median(), 3),
                    win_sc_min=round(_prof["scale_consistency"].min(), 3),
                    win_sc_max=round(_prof["scale_consistency"].max(), 3),
                    win_entropy_median=round(_prof["entropy"].median(), 2),
                    n_windows=len(_prof),
                )
            )
    window_results = pd.DataFrame(_rows)
    window_results
    return sample_low, sample_post, window_profiles


@app.cell
def _(pd, sample_low, sample_post):
    # Save pool for other notebooks to use
    post_tonal_like = pd.concat([sample_low, sample_post])[["title", "composer_name"]]
    post_tonal_like.to_pickle("./data/post-tonal-works.pkl")
    return


@app.cell
def _(np, plt, window_profiles):
    # One panel per piece: window entropy against bar number, post-tonal in red,
    # low-consistency comparison pieces in blue. Gaps are windows with too few notes.
    entropy_fig, _axes = plt.subplots(5, 2, figsize=(11, 12), sharey=True)
    _colors = {"post-tonal": "tab:red", "low-sc": "tab:blue"}
    # Column 0 holds the post-tonal pieces, column 1 the comparison pieces.
    for _col, _group in enumerate(["post-tonal", "low-sc"]):
        _items = [(k, p) for k, (g, p) in window_profiles.items() if g == _group]
        for _row, (_label, _prof) in enumerate(_items):
            _ax = _axes[_row, _col]
            _ax.plot(_prof["start_bar"], _prof["entropy"], color=_colors[_group])
            # Reference lines: a uniform spread over 7 notes (a full diatonic scale) and over all 12.
            _ax.axhline(np.log2(7), ls=":", c="gray", lw=0.8)
            _ax.axhline(np.log2(12), ls="--", c="gray", lw=0.8)
            _ax.set_title(_label, fontsize=8)
            _ax.set_xlabel("window start (bar)", fontsize=7)
            _ax.set_ylabel("entropy (bits)", fontsize=7)
    entropy_fig.suptitle("Pitch-class entropy over time (8-bar windows, hop 4)")
    entropy_fig.tight_layout()
    entropy_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Window consistency separates the groups:
     - Post-tonal pieces: They stay low in every window, with medians within 0.04 of the whole-piece value (Ligeti is 0.06 higher).
     - Comparison pieces: Their windows jump to a median of about 0.9, with maxima of 1.0. They're modulating or locally diatonic, and only the whole-piece number makes them look non-tonal.

     - The whole-piece number alone doesn't separate them. Show Yourself from Frozen 2 has 0.66 overall, which is lower than Berg (0.71) and Messiaen (0.76). The windowed version shows it is diatonic in each window, so this is where the window approach helps.

     - Entropy:
         - Webern, Berg and Schönberg: They sit at the near-flat ceiling of 3.4–3.55 bits (the maximum is 3.58) for almost the whole piece. This is the clearest signature.
         - Ligeti and Messiaen: They are lower, at 3.0–3.3 bits, which fits their modal or limited-transposition writing.
         - Comparison pieces: They hover around the 2.8-bit line (a full diatonic scale) and wander. Nuvida and Show Yourself show repeated dips, which look like key changes.

     - Caveats:
         - Tail drop: Every piece's last point falls sharply, for example Ligeti to 0.6 and Show Yourself to 1.7. That is the short final window, which holds only 1–2 bars of notes and so can't have high entropy. It's an artifact, not music. I'd  drop partial windows from the plot, or mark them.
         - Sample size: There are only five post-tonal pieces, and the comparison pieces
           were drawn at random. This is suggestive, not a test.
         - Digitaciones modos: It has a gap near bar 30, from windows with too few notes.
           Its entropy is flat at 2.9 bits, which may be a scale exercise.

     Conclusion: Whole-piece scale_consistency is a weak post-tonal detector, because
     modulating pieces get low scores too. Its windowed version, together with window
     entropy, separates these two groups cleanly in this small sample.
    """)
    return


if __name__ == "__main__":
    app.run()
