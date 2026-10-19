# Harmonic Boundary Detection in Symbolic Music via Hierarchical Clustering

## Abstract

This project studies whether harmonic boundaries, the points where functional harmony changes, can be recovered from pitch and voice-leading data without labeled training data, a fixed window size, or a hand-tuned threshold. We build a dendrogram over the notes of a piece and score each pair of temporally adjacent notes by its merge height, a resolution-independent measure of boundary strength. We evaluate boundary placement against expert chord-change annotations on the Bach chorale corpus, comparing against the neural hidden semi-Markov model (HSMM) of Uehara (2024). We also test whether the method extends to non-tonal repertoire, where no chord or key template applies. \[Results and final figures to be added.\]

## 1. Introduction

In tonal music, notes are organized around a key, and chords built from that key carry harmonic function. Real scores rarely state chord tones cleanly. Texture, ornamentation, and voice leading spread a chord's notes across voices and time, and measure boundaries often diverge from harmonic boundaries. Automatic harmonic segmentation, which recovers the underlying chord and key sequence and where it changes, is a long-standing problem in musical informatics.

Existing approaches fall into three groups:

- **Rule-based systems** (e.g. Melisma) depend on extensive hand-designed preference rules.
- **Supervised models** (semi-CRF, Harmony Transformer, CRNN+NADE) require labeled data and generalize poorly beyond their training distribution, mostly classical piano and vocal pieces.
- **Unsupervised models** (IGMM, HMM/HSMM) avoid labels but have structural limitations.

No ground-truth data on harmonic boundaries alone, independent of chord-quality labels, exists at scale. We therefore frame the task as unsupervised boundary discovery rather than classification.

The project has two aims:

1. **Unsupervised** boundary detection.
2. **Tonality-agnostic** boundary detection. Post-tonal music (Schoenberg, Webern, Berg), Messiaen's modes of limited transposition, Debussy's planing, and quartal or cluster voicings have no governing tonic and no closed chord vocabulary. Our clustering stage works on pitch-class co-occurrence rather than a chord or key template, so it does not presuppose tonal structure when finding boundaries. Only the labeling step does.

### Research Questions

- **RQ1.** Can harmonic boundaries be recovered from pitch and voice-leading data without a fixed window size or hand-tuned threshold, and do they achieve higher boundary F1 than fixed-window and rule-based baselines?
- **RQ2.** Does boundary quality hold on post-tonal and non-functional repertoire, where no chord or key template applies?

Label accuracy is reported as a secondary metric and is not a research question. Boundary stability across clustering resolution is a stated property of the method (see Section 4.2) but is not a separate research question.

## 2. Related Work

- **Raphael and Stoddard (2003).** HMM-based harmonic analysis. Chord-degree transitions were learned without supervision, but key transitions and post-modulation chord probabilities were hard to learn. \[citation to complete\]
- **Wang and Wechsler (2012).** Infinite Gaussian Mixture Model (IGMM) for key and chord recovery through Bayesian sampling. Gaussian profiles were set manually, and the model has no sequence component, so chord transitions are not modeled. \[citation to complete\]
- **Uehara (2024).** Neural HSMM with chord-quality templates. Jointly models keys and chord transitions and infers the number of modes dynamically. Strongest unsupervised result on Bach chorales (Full RN 61.6%, Root RN 66.9%). Limitations include no support for borrowed chords and over-labeling of passing tones. \[citation to complete\]
- **Supervised and rule-based systems.** Melisma, semi-CRF, Harmony Transformer, and CRNN+NADE. Summarized in Section 1 as background. \[citations to complete\]

Our contribution relative to this work is not raw accuracy against the HSMM. It is (a) boundary detection that needs no fixed resolution parameter and (b) an evaluation that does not assume tonality.

## 3. Datasets

| Corpus | Role | Contents |
| --- | --- | --- |
| PDMX (Zenodo, DOI 10.5281/zenodo.15571083) | Exploratory | 250K+ public-domain MusicXML scores from MuseScore, with composer, genre, licensing, and musical-feature metadata |
| Music21 Bach chorales | Validation | 371 chorales in MusicXML |
| Riemenschneider 1 to 20 subset | Annotated validation | Professional Roman-numeral annotations for 20 chorales, matching the HSMM evaluation setup |

Open decision: whether to replicate Uehara's exact 243/62/17 train/dev/test split for apples-to-apples comparison, or draw an independent split because our training corpus is PDMX rather than the chorale set alone. \[TBD\]

Raw audio and Spotify performance data are a deprioritized stretch goal.

## 4. Methodology

### 4.1 Featurization

Hand-crafted features per note:

- Pitch class
- Duration
- Onset (absolute position in piece)
- Melodic interval from previous note in the same voice
- Melodic contour direction (up, down, same) in the same voice
- Metric position (beat strength and position within measure)
- Local note density (notes per beat in a small window), a proxy for passing-tone and ornamentation likelihood
- Co-occurring pitch classes at the note's exact onset (12-dimensional presence or count vector across sounding voices)

Final feature set and normalization scheme are to be set after initial experimentation. \[TBD: include or cut metric position, contour, and note density\]

In parallel, we explore learned representations. Each note is encoded as the 512-dimensional hidden state of a pretrained Polyphony RNN (Magenta; Bach chorale checkpoint). The same clustering pipeline is then applied, and each cluster remains traceable to note onsets and song regions. Boundary discovery under this representation is compared against the hand-crafted baseline.

### 4.2 Hierarchical Clustering and Boundary Strength

We build a dendrogram over all notes rather than fixing a cluster count. For each pair of temporally adjacent notes, the boundary-strength score is the merge height: the level in the tree at which the two notes are first joined into one cluster.

- A low merge height means the notes are dissimilar and separate early, a strong boundary.
- A high merge height means they join late, a weak boundary.
- Because the score comes from the full dendrogram, no resolution parameter is imposed. Boundaries that persist across scale have high merge heights, which gives a stability measure without a separate experiment.

Open decisions: distance metric and linkage rule. \[TBD\]

### 4.3 Cluster Labeling and Evaluation

Boundary evaluation is the primary metric: precision, recall, and F1 of predicted boundaries against annotated chord-change onsets.

Label evaluation is secondary and proceeds in two steps:

1. For each note in a cluster, look up the ground-truth Roman-numeral label from the annotation.
2. Assign the cluster the majority label across its notes.

Because segment boundaries are not forced onto measure lines, algorithm output and annotations are resampled onto a common 16th-note timeline and compared label by label.

Accuracy is computed at every dendrogram cut and reported as a curve over cluster count. The cut at the largest jump in consecutive merge heights, a label-free heuristic, is reported as the headline result for comparison with the HSMM.

### 4.4 Tonal Sanity Check

The pipeline is run on the annotated Bach subset to confirm it recovers known harmonic structure. Boundary F1 and label accuracy are compared against Uehara (2024). The HSMM is used only here, because its architecture assumes tonality.

Open decisions: whether to retrain the HSMM on our split or use the pretrained checkpoint. \[TBD\]

### 4.5 Atonal Analysis

To test RQ2, segments are reduced to a sequence of 12-dimensional pitch-class vectors containing only the structurally important notes, analogous to an informal chord progression. The reduced sequence is compared with the original note-by-note pitch-class sequence using Dynamic Time Warping (DTW).

Importance is measured by ablation sensitivity:

1. Remove a note and recompute boundary detection in a local window (segment plus neighbors).
2. Measure the shift in local merge heights and boundaries.
3. Large shift means structural, so upweight. No shift means ornamental, so downweight.

The final comparison procedure:

1. Weight each note by duration, metric strength, and ablation sensitivity.
2. Accumulate into a normalized 12-dimensional pitch-class vector.
3. Compute DTW against the original sequence to get a similarity score.
4. Report the compression ratio (segments per note) alongside DTW to detect over-segmentation that inflates the score.
5. Apply the same procedure to the HSMM output for a comparable baseline.

Limitation: ablation sensitivity uses the clustering algorithm to decide which content is harmonically real, and the reduction is then used to evaluate that same algorithm. This circularity is acknowledged and is a point for discussion. \[Pending advisor input\]

### 4.6 Hypothesis Testing

We will test whether discovered boundaries align with annotated boundaries more than chance, using a correlation or permutation test adapted from the shuffle test in the original TDA proposal. Test design is not yet finalized. \[TBD\]

## 5. Experimental Setup

- Parsing and note sequences: music21 (including `chordify()` for co-occurrence context)
- Clustering: scikit-learn and SciPy hierarchical clustering
- Baseline: Uehara (2024) HSMM, `github.com/yui-u/harmonic-analysis-chorales` (Apache-2.0)
- Learned features: Magenta Polyphony RNN
- Deliverables: GitHub repository with cleaned data, Jupyter notebook, and technical report

## 6. Results

&#91;Stub: results to be filled in from the local analysis environment.\]

### 6.1 Tonal Sanity Check (Bach Chorales)

&#91;TBD\]

### 6.2 Boundary Detection Against Baselines (RQ1)

&#91;TBD\]

### 6.3 Atonal and Post-Tonal Analysis (RQ2)

&#91;TBD\]

### 6.4 Learned vs. Hand-Crafted Features

&#91;TBD\]

## 7. Discussion

&#91;TBD\]

## 8. Limitations

- Uehara's HSMM does not support borrowed chords and over-labels passing tones, so it is a strong but imperfect reference.
- The validation corpus is Bach chorales, which limits generality until the exploratory PDMX analysis is extended.
- Ablation-based importance weighting is circular with respect to the clustering method (Section 4.5).

## 9. Future Work

- Composer and genre variety beyond Bach chorales.
- Raw audio extension (stretch goal).
- Alternative clustering formulations, such as ToMATo persistence-based clustering, as a more formal version of the merge-height heuristic.

## References

&#91;To be formatted. Entries referenced in Sections 1 and 2: Raphael and Stoddard (2003); Wang and Wechsler (2012); Uehara (2024); Engel et al. (2017, Polyphony RNN / Magenta); Melisma; semi-CRF harmonic segmentation; Harmony Transformer; CRNN+NADE; PDMX (Zenodo 15571083); music21.\]
