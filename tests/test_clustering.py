import pytest


def validate_pipeline_outputs(selected_score):
    assert selected_score.metadata.title == "Aus meines Herzens Grunde"
    assert len(selected_score.parts) == 4
    assert len(note_features) == 229
    assert note_features["source_element"].notna().all()
    assert not any(element.duration.isGrace for element in note_features["source_element"])

    part_by_id = {str(part.id): part for part in selected_score_parts}
    source_keys = set()
    for _, row in note_features.iterrows():
        part = part_by_id[row["part_id"]]
        source_events = list(part.recurse().notes)
        event = source_events[int(row["source_event_index"])]
        assert event is row["source_element"]
        assert not event.duration.isGrace
        pitches = [event.pitch] if isinstance(event, Note) else list(event.pitches)
        pitch = pitches[int(row["source_pitch_index"])]
        assert row["pitch_name"] == pitch.nameWithOctave
        assert row["pitch_class"] == int(pitch.pitchClass)
        assert row["score_offset_ql"] == event.getOffsetInHierarchy(selected_score)
        assert row["duration_ql"] == event.duration.quarterLength
        source_key = (
            row["part_id"],
            int(row["source_event_index"]),
            int(row["source_pitch_index"]),
        )
        assert source_key not in source_keys
        source_keys.add(source_key)

    # The score begins with a one-quarter pickup; music21 identifies its onset as beat 3 of 3/4.
    first_onset = note_features["score_offset_ql"].min()
    pickup_rows = note_features.loc[note_features["score_offset_ql"] == first_onset]
    assert set(pickup_rows["pitch_name"]) == {"D4", "G2", "G4", "B3"}
    assert set(pickup_rows["beat_number"]) == {3.0}
    assert set(pickup_rows["beat_strength"]) == {"weak"}

    # Verify all 12 onset count vectors against the active note rows at every onset.
    for onset in note_features["score_offset_ql"].unique():
        active_rows = note_features.loc[(note_features["score_offset_ql"] <= onset) & (note_features["score_offset_ql"] + note_features["duration_ql"] > onset)]
        expected_counts = Counter(active_rows["pitch_class"].astype(int))
        onset_rows = note_features.loc[note_features["score_offset_ql"] == onset]
        for _, row in onset_rows.iterrows():
            actual_counts = {pitch_class: int(row[f"cooccur_pc_{pitch_class}_count"]) for pitch_class in range(12)}
            assert actual_counts == {pitch_class: expected_counts.get(pitch_class, 0) for pitch_class in range(12)}

    order_keys = list(
        zip(
            note_features["score_offset_ql"],
            note_features["part_id"],
            note_features["pitch_name"],
            note_features["source_event_index"],
            note_features["source_pitch_index"],
        )
    )
    assert order_keys == sorted(order_keys)

    assert linkage_matrix.shape == (len(note_features) - 1, 4)
    assert len(adjacent_merge_heights) == len(note_features) - 1
    assert np.isfinite(linkage_matrix).all()
    assert np.isfinite(adjacent_merge_heights["merge_height"]).all()
    assert np.all(np.diff(linkage_matrix[:, 2]) >= -1e-12)
    assert set(scaling_results) == {
        "unscaled_numeric",
        "minmax_numeric_columns",
        "minmax_time__normalized_cooccurrence",
    }
    assert all(len(result["adjacent_merge_heights"]) == len(note_features) - 1 for result in scaling_results.values())

    return {
        "score": selected_score.metadata.title,
        "parts": len(selected_score_parts),
        "note_rows_checked": len(note_features),
        "source_traceability": "passed",
        "onset_count_vectors": "passed for every onset",
        "pickup_beat_rule": "passed (beat 3 classified weak)",
        "dendrogram_merges": len(linkage_matrix),
        "adjacent_merge_heights": len(adjacent_merge_heights),
        "scaling_variants": len(scaling_results),
        "evaluation_performed": False,
    }

pipeline_validation = validate_pipeline_outputs()
print("All implementation checks passed.")