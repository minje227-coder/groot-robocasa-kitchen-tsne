# GR00T RoboCasa-Kitchen t-SNE

Interactive, manifest-aligned t-SNE viewer for GR00T N1.7 RoboCasa-Kitchen checkpoints and action representations.

Live site: <https://minje227-coder.github.io/groot-robocasa-kitchen-tsne/>

## Published features

| Run | Feature |
| --- | --- |
| Baseline 60K | raw VLM output, processed output |
| RKD UnitMu 60K | processed output |
| RKD RawMu W0.3 to 0 Cosine 60K | processed output |
| RKD RawMu Distance:Angle 1:1 60K | processed output |
| RKD RawMu Distance Only 60K | processed output |
| RKD RawMu Angle Only 60K | processed output |
| DTW-Sammon RawZ Angle Only | Processed-H + trained StudentHead output |
| RS-CL ParityState βs=1 (TestV1) | processed H |
| RS-CL JointShared βs,βa=1 (TestV1) | processed H |
| RS-CL JointShared βs=1,βa=0.3 (TestV1) | processed H |
| CSN StateOnly βs=1 (TestV1) | processed H |
| CSN JointSubspace βs,βa=1 (TestV1) | processed H + normalized projector Z + state/action-masked Z |
| CSN JointSubspace βs=1,βa=0.3 (TestV1) | processed H + normalized projector Z + state/action-masked Z |
| TimewarpVAE RawMu | action-only posterior `mu` |
| TimewarpVAE UnitMu | action-only L2-normalized `mu` |

All charts use the same `kitchen_24task_ep10_desc3_seed42_frame30_v1` manifest: 24 tasks, 10 episodes per task, and 30 uniformly sampled frames per episode (7,200 points total). The canonical source manifest SHA-256 is recorded in every run manifest.

## Embedding recipe

General feature matrices are independently filtered for constant dimensions, z-scored, reduced with PCA to at most 50 dimensions, then embedded with t-SNE using perplexity 30, 1,000 iterations, and seed 42.

CSN-only views follow the checkpoint's canonical path: appended SummaryToken output -> 128D projector -> L2 normalization -> static `ReLU(state/action mask)`, without post-mask renormalization. Their t-SNE preprocessing deliberately omits per-dimension z-scoring so learned mask magnitudes are not canceled; it uses constant filtering, centered PCA up to 50D, perplexity 30, 1,000 iterations, and seed 42. The model profile shows the 128D mask heatmaps, active/support overlap, RMS, effective rank, and mean sample norm.

The browser UI is shared with [`groot-insight-tsne`](https://github.com/minje227-coder/groot-insight-tsne), while manifests, task grouping, cameras, model profiles, features, and clips are RoboCasa-Kitchen-specific.

## Rebuild

DGX build helpers live in `scripts/`. Intermediate feature matrices are written below the ignored `cache/` directory. `scripts/validate_site.py` enforces manifest provenance, all run/point counts, sequence coverage, and all 720 clip files before publication.
