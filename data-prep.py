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


    return Path, mo, pd, tarfile, urllib


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

    DATA_DIR = Path("data/raw/pdmx")
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
        urllib.request.urlretrieve(MXL_URL, MXL_ARCHIVE)
    return


@app.cell
def _(METADATA_CSV, METADATA_CSV_URL, urllib):
    if not METADATA_CSV.exists():
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
    genre_df = genre_df[
        (genre_df["genres"] != "NA") & (genre_df["composer_name"] != "NA")
    ]

    return (genre_df,)


@app.cell
def top_composers_by_genre(genre_df):
    TARGET_GENRES = ["classical", "jazz", "rock", "metal"]

    top_composers_by_genre = {
        genre: (
            genre_df[genre_df["genres"] == genre]["composer_name"]
            .value_counts()
            .head(5)
        )
        for genre in TARGET_GENRES
    }
    top_composers_by_genre

    return


if __name__ == "__main__":
    app.run()
