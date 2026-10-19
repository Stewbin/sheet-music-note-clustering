import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from collections import Counter
    from math import isfinite
    import pandas as pd
    from music21.chord import Chord
    from music21.note import Note
    from music21.stream import Measure
    from music21.corpus import chorales
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import dendrogram
    import matplotlib.pyplot as plt


@app.cell(hide_code=True)
def _():
    from pathlib import Path
    from music21 import converter

    pdmx_root = Path("data/pdmx")
    pdmx_metadata_path = pdmx_root / "PDMX.csv"
    pdmx_sample_size = 10
    pdmx_sample_seed = 42  # Set to None to draw a new random sample on each run.

    # Use the deduplicated subset as the project's current working PDMX subset.
    pdmx_metadata = pd.read_csv(
        pdmx_metadata_path,
        usecols=["title", "mxl", "subset:deduplicated"],
    )
    pdmx_candidates = pdmx_metadata.loc[pdmx_metadata["subset:deduplicated"] & pdmx_metadata["title"].notna() & pdmx_metadata["mxl"].notna()]
    if len(pdmx_candidates) < pdmx_sample_size:
        raise ValueError(f"Only {len(pdmx_candidates)} eligible PDMX scores; cannot sample {pdmx_sample_size}.")

    sampled_pdmx_rows = pdmx_candidates.sample(
        n=pdmx_sample_size,
        random_state=pdmx_sample_seed,
    )
    selected_scores = []
    piece_summary_rows = []
    part_report_rows = []

    for pdmx_index, metadata_row in sampled_pdmx_rows.iterrows():
        # PDMX stores paths like ./mxl/1/11/....mxl relative to data/pdmx/.
        relative_mxl_path = Path(str(metadata_row["mxl"]).removeprefix("./"))
        score_path = pdmx_root / relative_mxl_path
        if not score_path.is_file():
            raise FileNotFoundError(f"PDMX score not found: {score_path}")

        score = converter.parse(score_path)
        parts = list(score.parts)
        selected_scores.append(score)
        pdmx_title = str(metadata_row["title"])
        pdmx_index = int(pdmx_index)  # Zero-based row index in PDMX.csv.
        piece_summary_rows.append(
            {
                "title": pdmx_title,
                "pdmx_index": pdmx_index,
                "part_count": len(parts),
            }
        )

        for part_number, part in enumerate(parts, start=1):
            part_name = getattr(part, "partName", None) or str(part.id)
            part_report_rows.append(
                {
                    "title": pdmx_title,
                    "pdmx_index": pdmx_index,
                    "part_number": part_number,
                    "part_id": str(part.id),
                    "part_name": str(part_name),
                    "note_events": len(part.recurse().notes),
                }
            )

    pdmx_piece_summary = pd.DataFrame(piece_summary_rows)
    pdmx_part_report = pd.DataFrame(part_report_rows)

    # Keep the existing single-score feature/clustering cells working on the first sample.
    selected_score = selected_scores[0]
    selected_score_parts = list(selected_score.parts)

    mo.vstack(
        [
            mo.md(
                f"""### Random PDMX sample
                Parsed {pdmx_sample_size} scores from the local `deduplicated` subset.
                `pdmx_index` is the zero-based row index in `PDMX.csv`; sample seed: {pdmx_sample_seed}.
                The first sampled score is passed to the existing single-score feature/clustering cells.
                """
            ),
            mo.md("#### Pieces"),
            pdmx_piece_summary,
            mo.md("#### Note events by part"),
            pdmx_part_report,
        ]
    )
    return sampled_pdmx_rows, selected_scores


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Featurize notes

    Each sampled score is featurized independently. One row represents one pitched note; chord pitches become separate rows, ties remain split, and grace notes are omitted. Roman-numeral annotations are excluded.

    - `pitch_class`: chromatic pitch class (0–11), with spelling retained separately; it is one-hot encoded for clustering.
    - `duration_ql`: note duration in quarter lengths.
    - `score_offset_ql`: absolute onset within that score, in quarter lengths.
    - `beat_number`: music21's onset beat; it may be `NaN` when the score has no meter.
    - `beat_strength`: music21's `note.beatStrength` value. If music21 returns a missing/non-finite value, use `0.5`.
    - `cooccur_pc_0_count` … `cooccur_pc_11_count`: counts of all notes sounding at the onset, including notes beginning then and counting doublings separately.
    - `source_part_id`, `source_event_index`, and `source_pitch_index` retain note-to-score traceability; `pdmx_index` and `title` identify the score.

    Melodic interval, contour, and local note density are deferred.
    """)
    return


@app.function(hide_code=True)
def featurize_handcrafted(score, parts):
    "Extract one row per pitched note, retaining its source Music21 event."
    rows = []

    for part in parts:
        part_id = str(part.id)
        part_name = getattr(part, "partName", None) or part_id

        for event_index, event in enumerate(part.recurse().notes):
            if event.duration.isGrace:
                continue

            if isinstance(event, Note):
                pitches = [event.pitch]
            elif isinstance(event, Chord):
                pitches = list(event.pitches)
            else:
                continue

            onset = event.getOffsetInHierarchy(score)
            duration = event.duration.quarterLength
            measure = event.getContextByClass(Measure)
            beat_number = float(event.beat)
            beat_strength = float(event.beatStrength)
            if not isfinite(beat_strength):
                beat_strength = 0.5

            for pitch_index, pitch in enumerate(pitches):
                rows.append(
                    {
                        "score_offset_ql": onset,
                        "duration_ql": duration,
                        "part_id": part_id,
                        "part_name": part_name,
                        "measure_number": measure.number if measure is not None else None,
                        "source_event_index": event_index,
                        "source_pitch_index": pitch_index,
                        "pitch_name": pitch.nameWithOctave,
                        "pitch_class_name": pitch.name,
                        "pitch_class": int(pitch.pitchClass),
                        "beat_number": beat_number,
                        "beat_strength": beat_strength,
                        "_end_offset_ql": onset + duration,
                        "source_element": event,
                    }
                )

    # Literal lexical part then pitch ordering resolves simultaneous onsets.
    # Source indices provide a stable final tie-break if those strings match.
    rows.sort(
        key=lambda row: (
            row["score_offset_ql"],
            row["part_id"],
            row["pitch_name"],
            row["source_event_index"],
            row["source_pitch_index"],
        )
    )

    pitch_class_columns = [f"cooccur_pc_{pc}_count" for pc in range(12)]
    onset_counts = {}
    for onset in sorted({row["score_offset_ql"] for row in rows}):
        sounding = [row for row in rows if row["score_offset_ql"] <= onset < row["_end_offset_ql"]]
        counts = Counter(row["pitch_class"] for row in sounding)
        onset_counts[onset] = {column: counts.get(pc, 0) for pc, column in enumerate(pitch_class_columns)}

    for row in rows:
        row.update(onset_counts[row["score_offset_ql"]])
        row.pop("_end_offset_ql")

    return pd.DataFrame(rows)


@app.cell(hide_code=True)
def _(cooccurrence_columns):
    def make_cluster_feature_matrix(features):
        """One-hot encode pitch class and append numeric feature columns."""
        pitch_class_columns = pd.DataFrame(
            {f"pitch_class_{pitch_class}": (features["pitch_class"].astype(int) == pitch_class).astype(float) for pitch_class in range(12)},
            index=features.index,
        )
        numeric_columns = [
            "duration_ql",
            "score_offset_ql",
            "beat_strength",
            *cooccurrence_columns,
        ]
        numeric_features = features[numeric_columns].astype(float)
        return pd.concat([pitch_class_columns, numeric_features], axis=1)

    return (make_cluster_feature_matrix,)


@app.cell(hide_code=True)
def featurize_the_scores(sampled_pdmx_rows, selected_scores):
    cooccurrence_columns = [f"cooccur_pc_{pc}_count" for pc in range(12)]

    # Keep one feature table per score: score offsets restart at zero, so these
    # tables must not be treated as one continuous note stream for clustering.
    note_features_by_pdmx_index = {}
    feature_frames = []
    _feature_summary_rows = []

    for (_pdmx_index, _metadata_row), _score in zip(
        sampled_pdmx_rows.iterrows(),
        selected_scores,
        strict=True,
    ):
        # add metadata to feature frame
        _pdmx_index = int(_pdmx_index)
        _parts = list(_score.parts)
        _features = featurize_handcrafted(_score, _parts)
        _features.insert(0, "pdmx_index", _pdmx_index)
        _features.insert(1, "title", str(_metadata_row["title"]))
        note_features_by_pdmx_index[_pdmx_index] = _features
        feature_frames.append(_features)

        # collect summary data of pieces
        _first_onset = _features["score_offset_ql"].min() if len(_features) else None
        _first_onset_total = (
            int(
                _features.loc[
                    _features["score_offset_ql"] == _first_onset,
                    cooccurrence_columns,
                ]
                .iloc[0]
                .sum()
            )
            if _first_onset is not None
            else 0
        )
        _feature_summary_rows.append(
            {
                "pdmx_index": _pdmx_index,
                "title": str(_metadata_row["title"]),
                "parts": len(_parts),
                "note_rows": len(_features),
                "note_rows_by_part": _features.groupby("part_id").size().to_dict(),
                "beat_strengths": _features["beat_strength"].value_counts().to_dict(),
                "cooccurrence_total_at_first_onset": _first_onset_total,
            }
        )

    all_note_features = pd.concat(feature_frames, ignore_index=True)
    feature_summary = pd.DataFrame(_feature_summary_rows)

    feature_preview_columns = [
        "pdmx_index",
        "title",
        "score_offset_ql",
        "part_id",
        "measure_number",
        "pitch_name",
        "pitch_class",
        "duration_ql",
        "beat_number",
        "beat_strength",
        *cooccurrence_columns,
    ]

    mo.vstack(
        [
            mo.md(
                f"""Featurized **{len(selected_scores)} sampled scores independently**.
                `note_features_by_pdmx_index` holds a separate feature table per PDMX row;
                `all_note_features` is a tagged, combined view for inspection. The legacy
                `note_features` variable remains the first sampled score for the current
                single-score clustering cells.
                """
            ),
            feature_summary,
            all_note_features[feature_preview_columns].head(12),
        ]
    )
    return cooccurrence_columns, feature_frames


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Build hierarchical clusters

    The first exploratory dendrogram uses cosine distance with average linkage. Pitch class is one-hot encoded; music21 `beat_strength`, duration, absolute score offset, and the 12 co-occurrence counts are numeric and currently unscaled. This is a starting representation for later scaling comparisons, not a selected final configuration.

    The table below reports, for each pair of consecutive note rows in the current score, the dendrogram height of their first common cluster. Rows are ordered by absolute onset, then literal lexical part and pitch; simultaneous-onset pairs are retained.
    """)
    return


@app.function(hide_code=True)
def lowest_common_merge_height(left_leaf, right_leaf, linkage_matrix, leaf_count):
    """Return the merge height where two leaves first share a cluster."""
    parents = [-1] * (2 * leaf_count - 1)
    heights = [0.0] * (2 * leaf_count - 1)
    for merge_index, (left, right, height, _) in enumerate(linkage_matrix):
        parent = leaf_count + merge_index
        parents[int(left)] = parent
        parents[int(right)] = parent
        heights[parent] = float(height)

    left_ancestors = set()
    node = int(left_leaf)
    while node != -1:
        left_ancestors.add(node)
        node = parents[node]

    node = int(right_leaf)
    while node not in left_ancestors:
        node = parents[node]
        if node == -1:
            raise ValueError("Leaves do not belong to the same dendrogram")
    return heights[node]


@app.cell
def run_clustering(feature_frames, make_cluster_feature_matrix):
    from scipy.sparse import csr_matrix, diags

    def build_temporal_connectivity(n_samples):
        """Connect consecutive score-wide note rows as a symmetric sparse chain."""
        if n_samples < 2:
            return csr_matrix((n_samples, n_samples), dtype=float)
        forward_edges = diags(
            np.ones(n_samples - 1, dtype=float),
            offsets=1,
            shape=(n_samples, n_samples),
            format="csr",
        )
        return (forward_edges + forward_edges.T).tocsr()

    def fit_full_tree_clustering(observations, connectivity):
        """Fit an average-linkage cosine tree constrained by score connectivity."""
        n_samples = observations.shape[0]
        if n_samples < 2:
            raise ValueError("Agglomerative clustering requires at least two note rows")
        if connectivity.shape != (n_samples, n_samples):
            raise ValueError(f"Connectivity shape {connectivity.shape} does not match {n_samples} observations")
        return AgglomerativeClustering(
            n_clusters=1,
            compute_full_tree=True,
            compute_distances=True,
            linkage="average",
            metric="cosine",
            connectivity=connectivity,
        ).fit(observations)

    models = []
    connectivities = []
    cluster_feature_matrices = []
    models_by_pdmx_index = {}
    connectivity_matrices_by_pdmx_index = {}
    cluster_feature_matrices_by_pdmx_index = {}
    _cluster_summary_rows = []

    for _frame in feature_frames:
        _pdmx_index = int(_frame["pdmx_index"].iloc[0])
        _title = str(_frame["title"].iloc[0])
        _n_samples = len(_frame)
        _feature_matrix = make_cluster_feature_matrix(_frame)
        _observations = _feature_matrix.to_numpy(dtype=float)

        if not np.isfinite(_observations).all():
            raise ValueError(f"PDMX index {_pdmx_index}: clustering features are non-finite")
        if np.any(np.linalg.norm(_observations, axis=1) == 0):
            raise ValueError(f"PDMX index {_pdmx_index}: cosine distance has a zero feature row")

        # Each frame already contains all parts, sorted score-wide by onset, part, and pitch.
        # The chain therefore includes cross-part neighbors but never joins different scores.
        _connectivity = build_temporal_connectivity(_n_samples)
        _expected_nnz = 2 * (_n_samples - 1)
        if _connectivity.nnz != _expected_nnz:
            raise ValueError(f"PDMX index {_pdmx_index}: unexpected temporal-chain edge count")
        if (_connectivity != _connectivity.T).nnz != 0:
            raise ValueError(f"PDMX index {_pdmx_index}: connectivity must be symmetric")
        if np.any(_connectivity.diagonal() != 0):
            raise ValueError(f"PDMX index {_pdmx_index}: connectivity must have a zero diagonal")

        _model = fit_full_tree_clustering(_observations, _connectivity)
        _distances = np.asarray(_model.distances_, dtype=float)
        _is_monotone = bool(np.all(np.diff(_distances) >= -1e-12))
        if _model.children_.shape != (_n_samples - 1, 2):
            raise ValueError(f"PDMX index {_pdmx_index}: sklearn did not build a full tree")
        if _distances.shape != (_n_samples - 1,) or not np.isfinite(_distances).all():
            raise ValueError(f"PDMX index {_pdmx_index}: invalid merge distances")
        if not _is_monotone:
            raise ValueError(f"PDMX index {_pdmx_index}: merge distances are not monotone")

        models.append(_model)
        connectivities.append(_connectivity)
        cluster_feature_matrices.append(_feature_matrix)
        models_by_pdmx_index[_pdmx_index] = _model
        connectivity_matrices_by_pdmx_index[_pdmx_index] = _connectivity
        cluster_feature_matrices_by_pdmx_index[_pdmx_index] = _feature_matrix
        _cluster_summary_rows.append(
            {
                "pdmx_index": _pdmx_index,
                "title": _title,
                "notes": _n_samples,
                "merges": len(_model.children_),
                "feature_dimensions": _feature_matrix.shape[1],
                "connectivity_nnz": _connectivity.nnz,
                "distances_monotone": _is_monotone,
            }
        )

    cluster_summary = pd.DataFrame(_cluster_summary_rows)
    mo.vstack(
        [
            mo.md("Fit one full sklearn dendrogram per score using a symmetric temporal-chain connectivity matrix."),
            cluster_summary,
        ]
    )
    return (
        cluster_feature_matrices_by_pdmx_index,
        connectivity_matrices_by_pdmx_index,
        fit_full_tree_clustering,
        models,
    )


@app.cell
def _():
    plt.savefig('figures/initial-dendrograms')
    return


@app.cell
def _():
    def sklearn_model_to_linkage_matrix(model):
        """Convert sklearn children_/distances_ to SciPy linkage format for plots."""
        n_samples = len(model.labels_)
        children = np.asarray(model.children_, dtype=int)
        distances = np.asarray(model.distances_, dtype=float)
        if children.shape != (n_samples - 1, 2) or distances.shape != (n_samples - 1,):
            raise ValueError("The sklearn model must contain a complete hierarchical tree")

        counts = np.zeros(n_samples - 1, dtype=float)
        for merge_index, merge in enumerate(children):
            current_count = 0
            for child_index in merge:
                if child_index < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_index - n_samples]
            counts[merge_index] = current_count

        return np.column_stack([children, distances, counts]).astype(float)

    def plot_dendrogram(model, **kwargs):
        """Plot a fitted sklearn model, converting its tree for SciPy's renderer."""
        return dendrogram(sklearn_model_to_linkage_matrix(model), **kwargs)

    return plot_dendrogram, sklearn_model_to_linkage_matrix


@app.cell
def plot_dendrograms(
    feature_frames,
    models,
    plot_dendrogram,
    sklearn_model_to_linkage_matrix,
):
    _limit = 15
    _n_models = len(models)
    if _n_models == 0:
        raise ValueError("No fitted clustering models are available to plot")

    _ncols = 2
    _nrows = int(np.ceil(_n_models / _ncols))
    _dendrogram_figure, _axes = plt.subplots(
        _nrows,
        _ncols,
        figsize=(16, 4.5 * _nrows),
        squeeze=False,
    )
    _axes_flat = _axes.ravel()
    linkage_matrices_by_pdmx_index = {}
    adjacent_merge_heights_by_pdmx_index = {}
    _adjacent_tables = []

    for _model_index, (_frame, _model) in enumerate(zip(feature_frames, models, strict=True)):
        _axis = _axes_flat[_model_index]
        _pdmx_index = int(_frame["pdmx_index"].iloc[0])
        _title = str(_frame["title"].iloc[0])
        _linkage_matrix = sklearn_model_to_linkage_matrix(_model)
        linkage_matrices_by_pdmx_index[_pdmx_index] = _linkage_matrix

        _leaf_order = plot_dendrogram(_model, no_plot=True)["leaves"]
        _labels = _frame["pitch_name"].iloc[_leaf_order].tolist()
        plot_dendrogram(
            _model,
            truncate_mode="level",
            p=_limit,
            labels=_labels,
            ax=_axis,
        )
        _axis.set_title(f"{_title} (PDMX {_pdmx_index})")
        _axis.set_xlabel("Notes")
        _axis.set_ylabel("Cosine distance")

        _adjacent_records = []
        for _left_index in range(len(_frame) - 1):
            _left = _frame.iloc[_left_index]
            _right = _frame.iloc[_left_index + 1]
            _adjacent_records.append(
                {
                    "pdmx_index": _pdmx_index,
                    "title": _title,
                    "left_row": _left_index,
                    "right_row": _left_index + 1,
                    "left_onset_ql": _left["score_offset_ql"],
                    "right_onset_ql": _right["score_offset_ql"],
                    "same_onset": _left["score_offset_ql"] == _right["score_offset_ql"],
                    "left_part": _left["part_id"],
                    "left_pitch": _left["pitch_name"],
                    "right_part": _right["part_id"],
                    "right_pitch": _right["pitch_name"],
                    "merge_height": lowest_common_merge_height(
                        _left_index,
                        _left_index + 1,
                        _linkage_matrix,
                        len(_frame),
                    ),
                }
            )
        _piece_adjacent_heights = pd.DataFrame(_adjacent_records)
        adjacent_merge_heights_by_pdmx_index[_pdmx_index] = _piece_adjacent_heights
        _adjacent_tables.append(_piece_adjacent_heights)

    for _axis in _axes_flat[_n_models:]:
        _axis.set_visible(False)

    _dendrogram_figure.tight_layout()
    adjacent_merge_heights = pd.concat(_adjacent_tables, ignore_index=True)
    mo.vstack([_dendrogram_figure, adjacent_merge_heights.head(20)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore merge height distrbution

    Bin the normalized merge heights of selected pieces.

    On average, the heights have a tangible right tail, so they should be log adjusted during evaluation.
    """)
    return


@app.cell
def _():
    n_bins = mo.ui.slider(label="Number of bins", start=1, stop=200, value=70)
    n_bins
    return (n_bins,)


@app.cell
def _(models, n_bins):
    dists = np.concatenate([m.distances_ / m.distances_.max() for m in models])
    plt.hist(dists, bins=n_bins.value)
    plt.ylabel("Counts")
    plt.xlabel("Normalized Heights")
    plt.title("Linear Adjustment")
    plt.show()
    print("Number of bins:", n_bins.value)
    plt.savefig('figures/initial-merge-heights')
    return


@app.cell
def _(models, n_bins):
    log_dists = np.concatenate([
        np.log(1 + m.distances_) / np.log(1 + m.distances_.max())
        for m in models
    ])
    plt.hist(log_dists, bins=n_bins.value)
    plt.ylabel("Counts")
    plt.xlabel("Normalized Heights")
    plt.title("Log Adjustment")
    plt.show()
    plt.savefig('figures/initial-log-merge-heights')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore feature scaling

    Compare three representations independently within each score, holding average linkage, cosine distance, and that score's temporal-chain connectivity matrix fixed:

    1. **Unscaled numeric values** (the initial clustering above).
    2. **Column-wise min–max:** scale duration, onset, music21 beat strength, and every co-occurrence count column to [0, 1]; leave pitch-class one-hot columns unchanged.
    3. **Min–max time/duration + row-normalized co-occurrence:** scale duration and onset to [0, 1], retain music21 beat strength in its native [0, 1] range, and divide each note's 12 pitch-class counts by their total sounding-note count; leave pitch-class one-hot columns unchanged.

    Each variant uses sklearn's full agglomerative tree and the same per-score connectivity matrix as the base representation. These are exploratory alternatives, not a final choice.
    """)
    return


@app.cell
def _(cluster_feature_matrices_by_pdmx_index, cooccurrence_columns):
    def minmax_columns(frame):
        """Scale columns to [0, 1]; constant columns become zero."""
        minima = frame.min(axis=0)
        ranges = frame.max(axis=0) - minima
        safe_ranges = ranges.mask(ranges == 0, 1.0)
        return (frame - minima) / safe_ranges

    _first_feature_matrix = next(iter(cluster_feature_matrices_by_pdmx_index.values()))
    categorical_feature_columns = [column for column in _first_feature_matrix.columns if column.startswith("pitch_class_")]
    numeric_feature_columns = [
        "duration_ql",
        "score_offset_ql",
        "beat_strength",
        *cooccurrence_columns,
    ]
    scaling_matrices_by_pdmx_index = {}

    for _pdmx_index, _base_matrix in cluster_feature_matrices_by_pdmx_index.items():
        _minmax_numeric = pd.concat(
            [
                _base_matrix[categorical_feature_columns],
                minmax_columns(_base_matrix[numeric_feature_columns]),
            ],
            axis=1,
        )
        _normalized_cooccurrence = _base_matrix[cooccurrence_columns].div(
            _base_matrix[cooccurrence_columns].sum(axis=1).replace(0, 1),
            axis=0,
        )
        _time_duration = minmax_columns(_base_matrix[["duration_ql", "score_offset_ql"]])
        _beat_strength = _base_matrix[["beat_strength"]]
        scaling_matrices_by_pdmx_index[_pdmx_index] = {
            "unscaled_numeric": _base_matrix.copy(),
            "minmax_numeric_columns": _minmax_numeric,
            "minmax_time__normalized_cooccurrence": pd.concat(
                [
                    _base_matrix[categorical_feature_columns],
                    _time_duration,
                    _beat_strength,
                    _normalized_cooccurrence,
                ],
                axis=1,
            ),
        }
    return (scaling_matrices_by_pdmx_index,)


@app.cell
def _(
    connectivity_matrices_by_pdmx_index,
    feature_frames,
    fit_full_tree_clustering,
    scaling_matrices_by_pdmx_index,
    sklearn_model_to_linkage_matrix,
):
    from scipy.stats import spearmanr

    scaling_results_by_pdmx_index = {}
    _scaling_summary_rows = []
    _scaling_correlation_rows = []
    _frames_by_pdmx_index = {int(_frame["pdmx_index"].iloc[0]): _frame for _frame in feature_frames}

    for _pdmx_index, _variant_matrices in scaling_matrices_by_pdmx_index.items():
        _frame = _frames_by_pdmx_index[_pdmx_index]
        _title = str(_frame["title"].iloc[0])
        _connectivity = connectivity_matrices_by_pdmx_index[_pdmx_index]
        _piece_results = {}

        for _variant_name, _feature_matrix in _variant_matrices.items():
            _observations = _feature_matrix.to_numpy(dtype=float)
            if not np.isfinite(_observations).all():
                raise ValueError(f"PDMX index {_pdmx_index}, {_variant_name}: non-finite features")
            if np.any(np.linalg.norm(_observations, axis=1) == 0):
                raise ValueError(f"PDMX index {_pdmx_index}, {_variant_name}: cosine has a zero row")

            _model = fit_full_tree_clustering(_observations, _connectivity)
            _linkage_matrix = sklearn_model_to_linkage_matrix(_model)
            _adjacent_heights = np.asarray(
                [lowest_common_merge_height(index, index + 1, _linkage_matrix, len(_frame)) for index in range(len(_frame) - 1)],
                dtype=float,
            )
            _piece_results[_variant_name] = {
                "feature_matrix": _feature_matrix,
                "model": _model,
                "linkage_matrix": _linkage_matrix,
                "adjacent_merge_heights": _adjacent_heights,
            }

        scaling_results_by_pdmx_index[_pdmx_index] = _piece_results
        _reference_heights = _piece_results["unscaled_numeric"]["adjacent_merge_heights"]
        for _variant_name, _result in _piece_results.items():
            _heights = _result["adjacent_merge_heights"]
            _scaling_summary_rows.append(
                {
                    "pdmx_index": _pdmx_index,
                    "title": _title,
                    "variant": _variant_name,
                    "feature_dimensions": _result["feature_matrix"].shape[1],
                    "min_merge_height": float(_heights.min()),
                    "median_merge_height": float(np.median(_heights)),
                    "max_merge_height": float(_heights.max()),
                    "zero_height_pairs": int(np.isclose(_heights, 0).sum()),
                    "spearman_vs_unscaled": float(spearmanr(_reference_heights, _heights).statistic),
                }
            )

        for _left_name, _left_result in _piece_results.items():
            for _right_name, _right_result in _piece_results.items():
                _scaling_correlation_rows.append(
                    {
                        "pdmx_index": _pdmx_index,
                        "title": _title,
                        "left_variant": _left_name,
                        "right_variant": _right_name,
                        "spearman_rho": float(
                            spearmanr(
                                _left_result["adjacent_merge_heights"],
                                _right_result["adjacent_merge_heights"],
                            ).statistic
                        ),
                    }
                )

    scaling_summary = pd.DataFrame(_scaling_summary_rows)
    scaling_rank_correlations = pd.DataFrame(_scaling_correlation_rows)
    mo.vstack(
        [
            mo.md("Scaling alternatives are fit per score using the same sklearn connectivity graph as the base model."),
            scaling_summary,
            scaling_rank_correlations,
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##
    """)
    return


if __name__ == "__main__":
    app.run()
