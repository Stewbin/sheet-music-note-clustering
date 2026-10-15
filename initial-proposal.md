# Proposal

> [!tldr;]
> Hierarchical clustering will be applied to discover harmonic boundaries in symbolic music, testing whether meaningful harmonic segmentation can be recovered from pitch/voice-leading data without labeled training data, and minimal unjustified hyperparameter choices.

---

## Motivation

In a piece, every note is assumed to be a part of a specific ordering called a *key*. *Chords* are groups of notes built from that key, each carrying a distinct harmonic function. And songs are then built from a sequence of chords, arranged in combinations to create tension and resolution. This is the rationale behind tonal music theory.

In real scores, chord tones are rarely stated cleanly: texture, ornamentation, and voice-leading scatter a chord's notes across voices and time, so the underlying chord is not directly readable from the notes alone. Even measure boundaries and harmonic boundaries frequently diverge. 

Automatic harmonic segmentation, recovering this underlying chord/key sequence and where it changes, is a popular point of research in musical informatics, but has traditionally lacked robust unsupervised solutions:

- **Rule-based systems** (e.g. Melisma) require extensive hand-designed preference rules.
- **Supervised models** (semi-CRF, Harmony Transformer, CRNN+NADE) require expensive labeled data and generalize poorly outside their training distribution (mostly piano/vocal classical pieces).
- **Unsupervised approaches** exist (IGMM, HMM/HSMM) but each has known structural limitations — see [[#Prior Work]].

No ground-truth data on harmonic boundaries alone (independent of chord-quality labels) exists at scale, which is why this is framed as an unsupervised discovery problem rather than a classification problem.

**Beyond common-practice tonality.** All of the systems above assume functional tonal harmony: a fixed tonic and a closed vocabulary of chord qualities. This breaks down for:

- **Post-tonal music** (Schoenberg, Webern, Berg) — no governing tonic; typically analyzed via pitch-class set theory instead of Roman numerals.
- **Other 20th-century idioms** — Messiaen's modes of limited transposition, Debussy's planing, quartal/cluster jazz voicings — organized around fixed collections or symmetric structures rather than tonic-dominant function.

Since this project's clustering stage operates on pitch-class co-occurrence rather than a fixed chord/key vocabulary, it does not presuppose tonal structure to find boundaries — only the labeling step does. 

Thus, this project has **two main aspirations**:
1. **unsupervised** boundary detection
2. **tonality-agnostic** boundary detection

## Research Questions

1. Can harmonic boundaries (where functional harmony changes) be discovered directly from pitch/voice-leading data, without a fixed window size or hand-tuned threshold?

2. Do discovered segment boundaries align with expert Roman-numeral annotations more closely than fixed-window or rule-based baselines?

3. Are the boundaries themselves more stable/meaningful than the labels — i.e., can harmonic change-points be found reliably even before the resulting segments can be labeled?

4. Can harmonic boundaries be detected in a tonality-agnostic way — i.e., does boundary quality (RQ1/RQ3) hold up on post-tonal/non-functional repertoire, where no chord/key template applies?

All RQs (research questions) are being pursued together, not sequentially.

## Prior Work

Past breakthroughs in unsupervised harmonic segmentation include:

- **Raphael & Stoddard (2003)** — HMM-based harmonic analysis; learned chord-degree transitions unsupervised, but key transitions and post-modulation chord probabilities were difficult to learn.
- **Wang & Wechsler (2012)** — Infinite Gaussian Mixture Model (IGMM); recovered keys/chords via Bayesian sampling, but used manually-set Gaussian profiles and could not model chord transitions (pure note-clustering, no sequence component).
- **Uehara (2024)** — Neural HSMM with chord-quality templates; jointly models keys and chord transitions and discovers the number of modes dynamically. Strongest unsupervised result on Bach chorales (Full RN 61.6%, Root RN 66.9%). Limitations: no support for borrowed chords, over-labels passing tones.

Although our proposed model likely loses in robustness to Uehara (2024), the project's proposed contribution, is not raw accuracy against HSMM, but:

- **Category-count discovery** — determining the number of harmonic categories from the data rather than assuming a fixed template set or state count.
- **Boundary stability across resolution (RQ3)** — studying whether harmonic boundaries are recoverable and stable across clustering granularity (via dendrogram merge height / persistence), a question not naturally addressed by a single Viterbi-optimal HSMM segmentation.

## Proposed Method

Our primary method is **hierarchical clustering**. We build a dendrogram over all notes rather than committing to a fixed number of clusters upfront, then use each pair of temporally adjacent notes' **merge height** — the level in the tree at which the two are first joined into the same cluster — as our boundary-strength score.

- **Resolution-independent boundary strength** — a low merge height means the notes are dissimilar and split apart even at fine resolution (a strong/early boundary); a high merge height means they're only unified late (a weak boundary).
- **No fixed-resolution parameter** — the measure comes from the full dendrogram, not one chosen cluster count, so it avoids imposing the fixed-resolution parameter RQ1 is meant to test the absence of.
- **Gives us RQ3's stability measure for free** — a boundary stable "across scale" is one with a high merge height, i.e. it persists across resolutions rather than being an artifact of one.

%%### Follow Up: ToMATo (Topological Mode Analysis Tool)

Tomato is a persistence-based clustering algorithm (Chazal et al., 2013) that identifies clusters as basins of attraction around density peaks, using persistent homology to compute a persistence diagram of those peaks (birth = peak prominence, death = the density level at which it merges into a more prominent neighboring peak).

Motivations for its use:
- **Cluster count read off the data** — the number of clusters comes from the largest gap in prominence values in the persistence diagram, rather than being supplied as a hyperparameter.
- **Built-in noise rejection** — low-persistence peaks can be treated as noise rather than real structure.
- **More principled version of the same question** — it answers the same boundary-persistence question hierarchical clustering is used for first, but with a formal, theoretically-grounded persistence measure instead of a merge-height heuristic.
%%
### Baselines

> **HSMM (Uehara, 2024)** — neural hidden semi-Markov model with chord quality templates; current strongest unsupervised harmonic analysis result. Pretrained checkpoints available (Apache-2.0) at `github.com/yui-u/harmonic-analysis-chorales`, trained on the same Music21 Bach chorale corpus. Used as the strong upper-bound comparison. 
> 
> Labels each note with (key, root, chordal quality).

### Cluster Labeling & Evaluation

 Since hierarchical clustering produces many possible clusterings:
 
- Labeling accuracy is computed across **every possible dendrogram cut** and reported as a curve over cut height/cluster count.
- But the cut at the **single biggest jump** in consecutive merge heights (an unsupervised, label-free heuristic) is additionally reported as the "most concrete", for headline comparison to the HSMM baseline.

#### Atonal Analysis

To keep in theme with the tonality agnostic analysis, we reduce the computed segments into a series of pitch-class vectors containing only the most "important" notes (analogous to a progression of informal chords). The resulting harmonically reduced piece is then compared to the original via Dynamic Time Warping (DTW). 

 Importance is determined via ablation sensitivity: 
 1. Each note is removed, and boundary detection is recomputed in a local window around it (segment + neighbors)
 2. The resulting shift in local merge heights/boundaries is measured
 3. Big shift → structurally important, upweight. No shift → ornamental, downweight.
 
 This assumption — that ablation sensitivity is a valid proxy for structural importance — is what makes the method tonality-agnostic, but it does introduce circularity: the clustering algorithm is used to decide what counts as "real" harmonic content, and that reduction is then used to evaluate the same algorithm's segmentation. 

> Q: Prof., is this a problem...?

The final comparison workflow looks like: 
1. Weight each note by duration × metric strength × ablation-sensitivity.
2. Accumulate into a 12-dim pitch-class vector, normalize.
3. DTW against the original note-by-note pitch-class sequence → similarity score.
4. Report compression ratio (segments / notes) alongside it, to catch over-segmentation gaming the DTW score.
5. Apply the same procedure to Uehara's HSMM output for a comparable baseline number.

#### Tonal Analysis

As a sanity check, the algorithms will be used to do Roman numeral analysis on a [[#^767565|validation corpus]] of professionally annotated tonal music (composed by Bach). Accuracy will be computed against the annotations, and compared against Uehara (2024)'s HSMM.

> [!NOTE] Remark: Baseline only used here
> Since Uehara (2024)'s model architecture specifically assumes tonality, it will not be used as a baseline in the atonal analysis. It is just for assuring that the algorithm can capture true harmonic pattern.

Cluster labeling proceeds as such:
1. For each note in a cluster, look up the ground-truth Roman-numeral/chord label that note actually belongs to (from the annotation)
2. Take the majority vote across the cluster's notes, and that's the cluster's assigned label.

Since we don't impose that segment boundaries align to measures, algorithm output and Roman-numeral annotations are resampled onto a common fine-grained timeline (e.g. every 16th note) and compared label-by-label, rather than measure-to-measure.
## Featurization

### Hand Crafted Scheme

**Feature list (per note):**
- Pitch class
- Duration
- Onset (absolute position in piece)
- Melodic interval from previous note (same voice)
- Melodic contour direction (up/down/same from previous note, same voice)
- Metric position (beat strength / position within measure)
- Local note density (notes per beat in a small window) — proxy for passing-tone/ornamentation likelihood
- Co-occurring pitch classes at this note's exact onset (12-dim presence/count vector across all sounding voices, flattened alongside other features)

### Learned Representations (Polyphony RNN)

In parallel, we will explore note-level representations learned via Google Magenta's Polyphony RNN (Engel et al., 2017). Rather than hand-crafting features, we encode each note as the hidden state vector from a pre-trained RNN encoder that has learned polyphonic music structure. This provides an alternative featurization strategy that may better capture implicit harmonic relationships compared to engineered features.

> Engel et al., "MusicVAE: Creating a Universal Music VAE with the Transformer VAE" (2017); Polyphony RNN documentation and pre-trained models at https://github.com/magenta/magenta/tree/main/magenta/models/polyphony_rnn.

**Approach:**
- Encode note sequences using Polyphony RNN hidden states (dimension: 512, from Bach chorale pre-trained checkpoint)
- Use identical clustering pipeline on the learned representation
- Maintain full temporal traceability: each cluster can still be mapped back to specific note onsets and song regions
- Compare boundary discovery (RQ1–RQ4) outcomes against the hand-crafted feature baseline
## Datasets

**Exploratory corpus:** [PDMX](https://zenodo.org/records/15571083) (DOI: 10.5281/zenodo.15571083) — 250K+ public domain MusicXML scores from MuseScore, with composer/genre/licensing/musical-feature metadata.

**Validation corpus**: Bach chorales from the **Music21 Library** (371 chorales, MusicXML), with professional Roman-numeral annotations for a 20-piece subset (Riemenschneider numbers 1–20), consistent with Uehara's HSMM evaluation setup.^767565

## Deliverables

- GitHub repository with:
	- Cleaned datasets (MusicXML + annotation files)
	- Jupyter notebook: full analysis + visualizations
	- Technical report: methods, results, interpretation
- Figures: cluster/segment visualizations, boundary-alignment comparisons against HSMM baseline, correlation/agreement tables

---

# Glossary

| Tool | Purpose |
| --- | --- |
| `music21` | Parse MusicXML; extract note sequences and co-occurrence context (`chordify()`); access Bach chorale corpus + annotations |
| scikit-learn / scipy | Hierarchical clustering implementation |
| GUDHI / `tomaster` | ToMATo persistence-based clustering implementation |
| Polyphony RNN (Magenta) | Pre-trained RNN encoder for learning note-level representations from polyphonic music (Bach chorale checkpoint) |
| Uehara HSMM repo | Pretrained/retrainable neural HSMM baseline (`github.com/yui-u/harmonic-analysis-chorales`) |
