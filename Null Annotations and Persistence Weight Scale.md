# Ground Truth Evaluation Protocol

## Gist

Correctness of predicted harmonic transition *placements* are first evaluated, by scoring the resultant dendrograms of pieces from a professionally annotated dataset against their true chord-change onsets (locations). Only the position is used, never the label.

A range of F1-measures are computed for each dendrogram for sensitivity analysis. An aggregate statistic computed from them is returned at the end.

## Step 1. Decide on tolerance set

1. Compute the distribution of gaps between consecutive annotated onsets.
2. Use tolerances exact (same 16th), ±1 eighth, and one larger value capped below half the typical annotated gap.
3. Drop ±1 beat if chorale harmonic rhythm is about one chord per beat.

> Why: a tolerance above half the gap lets one prediction match a neighboring onset, which inflates scores.

## Step 2. Run the algorithm

Cluster notes with **connectivity-constrained agglomerative clustering** (chain graph over temporally adjacent notes), so every segment is contiguous.

A **gap** is the junction between two temporally adjacent notes, i.e. a candidate boundary. The **boundary strength h\_i** of gap i is the height at which the segments on either side of it merge.

## **Step** 3. Match predictions to annotations

Boundary placement is treated as a classification problem in which **every gap is a prediction**, weighted by its persistence (Step 4). A prediction is considered **correct iff it falls within some tolerance of a true onset not already matched to another prediction**.

- **TP:** matched transitions.&#32;
- **FP:** unmatched transitions.&#32;
- **FN:** unmatched annotated onsets.

Boundaries are matched **one-to-one**: each annotated onset matches at most one prediction, and each prediction at most one onset. Candidates within tolerance are matched nearest first.

> **Why:** scoring positions only avoids the majority-vote problem, where accuracy rises with cluster count. One-to-one matching stops several predictions near one onset from all counting as correct, so over-segmentation lowers precision.

## Step 4. Persistence-weighted scores

When calculating the confusion matrix statistics (TP, FP, FN), correct boundary placements are weighted by how long they live, i.e. the height in the dendrogram right before it disappears:

```latex
w_i={h_i\over h_\text{max}}
```

Let M be the matched predictions, U the unmatched predictions, and N the number of annotated onsets. Then the weighted confusion statistics are:

```latex
TP_w = \sum_{i \in M} w_i \qquad FP_w=\sum_{i \in U} w_i \qquad FN_w=N-TP_w 
```

> **Why:** a boundary that is correct only briefly in the dendrogram earns little credit, and a short-lived spurious boundary costs little. This also gives the boundary-stability contribution an evaluation.

Weighted precision, recall, and F1-measure are:

```latex
P_w = \frac{TP_w}{TP_w + FP_w}, \qquad R_w = \frac{TP_w}{TP_w + FN_w}, \qquad F_{w} = \frac{2P_wR_w}{P_w + R_w}
```

## Null Experiments

| Null Hypothesis | Experiment Setup |
| --- | --- |
| Cluster lifetime is uninformative | Permute strengths among beat-aligned gaps, keeping their distribution |
| Boundary alignments occur by chance | Rotate the annotated onsets in time |
| Clusters form based on local statistics, not harmonic structure | 1. Run on measure-shuffled music, score against the **original** annotations<br>2. Run on random-note music (matched pitch and rhythm statistics), score against the annotations of a real piece of the same length |

For each H\_0, and tolerance, a p-value is computed

```latex
p=\frac{1+\#\{\mathrm{F_{null}\geq\mathrm{F_{observed}}\}}}{B+1}
```

- F\_observed = mean of F-measures from annotated Bach Chorales with a given tolerance
- B = number of null replicates under H\_0, each a full run over the 20 chorales aggregated the same way as F\_observed&#32;

The formula comes from assuming all F are equally likely to be seen under H\_0.

## Baselines

1. **U****e****h****ara (202****4****)****'****s** **HSMM**
   1. boundaries taken from its chord-change positions
   2. evaluated against unweighted F-measure
2. Our algorithm: dendrogram cut at the largest jump between consecutive merge heights, evaluated against unweighted F-measure

Score every baseline at **every tolerance** from Step 1.

> **Why:** F scores rise as tolerance widens, so methods are only comparable at the same tolerance.
