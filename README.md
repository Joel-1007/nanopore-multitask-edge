# Edge-Deployable Multi-Task Nanopore Deep Learning Framework

Group 13 · DNA Sequencing Technologies · Amrita Vishwa Vidyapeetham, Coimbatore
Team: Aditya, Joel, Dinesh, Ansh

A compact CNN–Transformer student that reads raw nanopore signal and jointly predicts the base sequence (CTC) and 6mA / 4mC / 5mC modification probabilities, trained on Dorado SUP v5.2.0 pseudo-labels. The long-term goal is edge deployment on a Jetson Orin Nano.

## Status (12 Oct 2026)
- Data ready: NO-MISS files 0–2 (6.15M windows) for training, and *E. faecium* EF26 (2.16M windows) held out as an external test set.
- Best student (930K params): 21.73% window CER against Dorado; 5mC AP 0.65; 4mC is not yet detected.
- FP16 export halves the weights file with no change in quality.
- EF26 evaluation, full-read inference, INT8/TensorRT and Jetson deployment are not done yet.

All metrics measure agreement with Dorado, not biological accuracy.

## Where to look
| File | Contents |
|---|---|
| [project_summary.md](project_summary.md) | Scope, datasets, results, roadmap and open work |
| [deep_analysis.md](deep_analysis.md) | Detailed numbers for basecalling, preprocessing, training and evaluation |
| [dinesh_run_instructions.md](dinesh_run_instructions.md) | Next steps on the Linux workstation |
| `JADA FULL RESULTS.pdf` | Full 52-page results record (source of truth) |
| [Nanopore_Numbered_Figures/](Nanopore_Numbered_Figures/) | All 41 report figures with captions (`index.html`) |
| `phase1_preprocessing.py`, `phase1_dataloader.py` | Early drafts; not the v0.2.0 code that built the datasets |

Raw data, BAMs, shards and checkpoints live on the workstation and are not in this repo.
