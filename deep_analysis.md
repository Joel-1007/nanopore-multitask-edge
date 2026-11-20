# Deep Analysis — Edge-Deployable Multi-Task Nanopore Framework
**Team:** Aditya · Joel · Dinesh · Ansh — Amrita Vishwa Vidyapeetham, Coimbatore  
**Last Updated:** 2026-10-12  
**Source of truth:** `JADA FULL RESULTS.pdf` (12 Oct 2026). Figure numbers below refer to `Nanopore_Numbered_Figures/`.  
**Phase:** Data preparation complete; student baseline and wider student trained and evaluated against Dorado; training frozen for review.

---

## 1. Executive Status

| Item | State | Key number |
|---|---|---|
| NO-MISS 0 basecall + preprocess | ✅ Complete, verified | 2,048,292 windows |
| NO-MISS 1 basecall + preprocess | ✅ Complete, verified | 2,048,181 windows |
| NO-MISS 2 basecall + preprocess | ✅ Complete, verified | 2,052,140 windows |
| NO-MISS 3–7 | ⛔ **Out of scope** (deliberately excluded) | — |
| EF26 (SRR32250932) re-basecall with moves + preprocess | ✅ Complete, verified, test-only | 2,155,099 windows |
| Student width 96 | ✅ Trained 50K steps | 29.42% CER |
| Student width 192 | ✅ Trained 100K steps (selected) | **21.73% CER** |
| FP16 export | ✅ Measured | −49.86% file size |
| EF26 external evaluation | ⬜ Not run | — |
| INT8 / TensorRT / Jetson | ⬜ Not started | — |

> [!IMPORTANT]
> All accuracy numbers are **agreement with Dorado pseudo-labels**, not biological accuracy. "100% − CER" must not be reported as accuracy.

---

## 2. Corrections to the Previous Version (26 Sep)

| Earlier claim | Corrected (from results) |
|---|---|
| Sampling rate 4 kHz, ~5 samples/base | **5,000 Hz**; Dorado move stride is **6** |
| Signal slice `[ts : ts+ns]` | `ns` counts samples **before trimming**, so the called segment is **`[ts, ns)`** (e.g. example 1 uses `[1,534, 44,992)` of a 44,993-sample parent) |
| ~10 kb average NO-MISS read | Median 668–842 bases, N50 3,778–4,052 bases |
| NO-MISS 0 has 4,442 split reads | **9,798 split-child records** (they are primary records, not supplementary) |
| PRJNA BAM missing move tables; +498 delta unexplained | EF26 re-basecalled with moves; all 124,016 records have all 5 tags. The +498 records are split children (995 records carry a parent-ID tag), with 0 secondary/supplementary |
| Files 1–7 pending | Files 1–2 done; 3–7 excluded from scope |
| Head B = 2 channels (5mC, 6mA) | **3 channels: 6mA, 4mC, 5mC** |
| ML probability = byte / 255 | Bin midpoint **(byte + 0.5) / 256** |

---

## 3. Basecalling — NO-MISS 0–2

All three are file partitions of run `853a956f-8add-4003-b3c4-7dafb39b6201`, flow cell PBK98658. Dorado SUP v5.2.0 + 6mA v1 + 4mC_5mC v1, move tables emitted, modification threshold 0.

| Measure | File 0 | File 1 | File 2 |
|---|---|---|---|
| Dorado version | 2.1.0 | 2.1.0 | 2.1.2 |
| Input parents | 304,990 | 281,112 | 267,271 |
| Represented parents | 304,551 | 280,782 | 266,985 |
| Missing parents | 439 | 330 | 286 |
| BAM records | 309,432 | 286,315 | 272,684 |
| Split-child records | 9,798 | 10,826 | 11,101 |
| Called bases | 480,009,459 | 483,921,143 | 485,132,837 |
| Median / N50 / max length | 668 / 3,778 / 44,840 | 761 / 3,979 / 48,883 | 842 / 4,052 / 43,276 |
| Mean predicted read Q | 22.31 | 22.57 | 22.76 |
| Required tags | 100% | 100% | 100% |
| Logged duration | 9 h 29 m 46 s (resumed only) | 32 m 18 s (resumed only) | 63 m 37 s (fresh) |
| Samples/s (logged) | 136,241 | 1,717,392 | 1,750,922 |

These timings are **not** a controlled version or hardware comparison. Files 0 and 1 include records recovered from earlier interrupted runs, and the batching differs between runs. The causes of the missing parents can't be determined from counts alone.

### Modification score summary (Dorado, BAM level)

| File | 6mA mean P / % ≥0.5 | 4mC mean P / % ≥0.5 | 5mC mean P / % ≥0.5 |
|---|---|---|---|
| 0 | 0.0771 / 3.215% | 0.0266 / 0.415% | 0.0134 / 0.682% |
| 1 | 0.0764 / 3.172% | 0.0266 / 0.418% | 0.0132 / 0.681% |
| 2 | 0.0766 / 3.190% | 0.0267 / 0.423% | 0.0132 / 0.681% |

Every eligible A has a 6mA score and every eligible C has 4mC and 5mC scores. G/T positions are **ineligible and must be masked**; their absence is not a negative label. Positives are rare (≤3.2%), which drives the class-imbalance problems in Section 7.

---

## 4. EF26 External Dataset (SRR32250932)

| Property | Value |
|---|---|
| Organism / isolate | *Enterococcus faecium* EF26 (accession table, Sci. Data 2025) |
| Flow cell / kit | FAW83592 / FLO-MIN114, SQK-NBD114-24, 5 kHz |
| FAST5 ↔ POD5 read IDs | 123,518 / 123,518 matched, 0 missing |
| Dorado run | 2.1.2, batch 32 / mod batch 128, resumed; 64 m 45 s; 1,744,079 samples/s |
| BAM audit (8 Oct) | 123,518/123,518 parents represented; 124,016 unique records; 0 secondary/supplementary; all 5 tags in every record |
| Build | 2,155,099 test windows from 123,514 parents; 51 shards; 81 m 56 s |
| Exclusions | 4 parents >2,000,000 samples; 9 empty-CTC windows |

Outstanding: repository checksum verification, EF26 ↔ NO-MISS disjointness check (so far verified within EF26 only), a verified reference genome, and independent modification labels.

---

## 5. Preprocessing (v0.2.0) — NO-MISS 2 Detail

* Normalization: `x_norm = (x − median) / (1.4826 × MAD)` over the **whole called segment** before windowing. Example 1: center 98.40 pA, scale 29.68 pA.
* Windows: 4,096 samples, hop 3,072 (1,024 overlap). Short final windows are zero-padded and get a validity mask.
* Labels per window: CTC target, soft modification probabilities and eligibility masks.

| Split | Parents | Windows | Padded windows | Shards |
|---|---|---|---|---|
| Train | 186,727 | 1,433,742 | 190,653 | 35 |
| Val | 40,139 | 307,925 | 41,032 | 8 |
| Test | 40,118 | 310,473 | 40,987 | 8 |

Exclusions: 286 parents without a BAM record; 12 empty-CTC windows; 1 window CTC-incompatible at encoder stride 4; 3,404 exported windows with no modification target. All shard and index hashes passed, there is zero parent overlap across splits, and 128 windows were spot-revalidated (a sample, not a full audit).

---

## 6. Student Model

```
signal (B×4096) ─ Conv1d k5 s2 → LN → GELU ─ Conv1d k5 s2 → LN → GELU   (stride 4 → B×1024×W)
               ─ sinusoidal pos ─ 2× pre-norm Transformer (4 heads, FF 3W, dropout 0.1) ─ LN
               ├─ Linear W→5  : blank/A/C/G/T  → CTC (greedy decode)
               └─ Linear W→3  : 6mA/4mC/5mC    → masked soft-target BCE at teacher move positions ÷ 4
```

| | Width 96 | Width 192 |
|---|---|---|
| Parameters | 234,728 | 930,248 |
| Steps / evaluated ckpt | 50,000 / 49,000 | 100,000 / 96,000 |
| Run dir | `runs/student_baseline_pilot_20261008T075006_726080466Z` | `student_w192_20261009T131226_647430928Z` |

### Experiment chronology (fixed 256-window validation sample)

| Experiment | CER |
|---|---|
| W96 pilot, 500 steps | 48.32% |
| W96, 5,000 steps | 36.92% |
| W96 weighted-mod trial (positives ×10), 5,000 steps | 37.21% (higher BCE; no matched AP diagnostic) |
| 8-window overfit diagnostic | 0% (memorization check only) |
| W96, 50,000 steps | 28.37% |
| W192, 50,000 steps | 22.76% |
| W192, 100,000 steps | 21.57% |

---

## 7. Evaluation (4,096 NO-MISS val windows, seed 123; 1,247,510 teacher bases)

### Sequence

| Metric | W96 @49k | W192 @96k |
|---|---|---|
| Window CER | 29.42% | **21.73%** |
| CTC loss | 0.758 | 0.575 |
| Mod BCE | 0.145 | 0.139 |

W192 error breakdown: 102,141 substitutions, 31,854 insertions, 137,094 deletions (**50.6% of edits are deletions**). The student emits 105,240 fewer bases than the teacher. 46 windows give empty predictions (4.55% of deletions), e.g. a GAATG-repeat window. Full windows: 21.56% CER; padded: 23.54%.

### Modifications (teacher P ≥ 0.5 = positive; student threshold 0.5)

| Channel | Positives | AP (W96 → W192) | Precision | Recall | TP / FP / FN |
|---|---|---|---|---|---|
| 6mA | 9,932 / 312,049 | 0.139 → 0.264 | 63.66% | 9.67% | 960 / 548 / 8,972 |
| 4mC | 1,269 / 308,830 | 0.009 → 0.011 | undefined | **0%** | 0 / 0 / 1,269 |
| 5mC | 2,154 / 308,830 | 0.264 → **0.651** | 73.60% | 54.74% | 1,179 / 423 / 975 |

Threshold trade-off (W192): at 0.1, 6mA gets 76% recall at 9% precision and 5mC gets 79% recall at 39% precision. 4mC still fails (3.6% recall, 1.6% precision). On teacher-confident (≥0.9) sites, mean student output is 0.29 (6mA), 0.04 (4mC), 0.57 (5mC), so the model **systematically underpredicts**. MAE is misleading here: always predicting zero beats W96 on 5mC MAE.

### Compression, decoding, timing

| Measurement | Result |
|---|---|
| FP32 → FP16 weights | 4,520,439 → 2,266,743 bytes (−49.86%); CER 21.7304% → 21.7291%; AP unchanged |
| Forward (batch 1, warm, AMP) | FP32-stored 2.20 ms / FP16-stored 2.15 ms median; peak alloc 20.0 → 16.1 MiB |
| Greedy vs beam-5 vs beam-10 (256 windows) | 21.35% / 20.32% / 20.26% CER; 0.10 / 46.8 / 93.2 ms CPU per window |
| Dorado on 100 val parents (2.83M samples) | 5.88 s wall, 1.60 s logged processing; 104 records, 214,264 bases |

Forward timings exclude I/O, normalization, transfer, decoding and stitching. There is **no valid student vs Dorado speedup** yet: the student side of the 100-read pilot hasn't run, and the student outputs windows while Dorado outputs whole reads.

---

## 8. What Can and Cannot Be Claimed

**Can claim:** a verified data pipeline (raw signal → normalized, labelled, leakage-checked shards), functional joint training, a 29.42% → 21.73% CER improvement with the wider model, and a 1.994× smaller FP16 file with unchanged agreement.

**Cannot claim:** biological accuracy, external generalization (EF26 unevaluated), calibrated modification probabilities, working 4mC detection, full-read output, or any speed or energy advantage over Dorado. INT8 and pruning are untested.

---

## 9. Open Issues

| # | Issue | Severity |
|---|---|---|
| 1 | EF26 external evaluation not run; eval command rejects a non-training manifest | 🔴 High |
| 2 | No verified EF26 reference or independent modification labels | 🔴 High |
| 3 | 4mC head makes no positive calls | 🔴 High |
| 4 | Mod head trained at teacher move positions, read at student-decoded positions | 🟡 Medium |
| 5 | No full-read stitching / coordinate reconstruction | 🟡 Medium |
| 6 | Paired Dorado vs student 100-read benchmark incomplete; no power/VRAM traces | 🟡 Medium |
| 7 | EF26 ↔ NO-MISS read-ID disjointness not verified | 🟡 Medium |
| 8 | Deletion-dominated sequence errors (50.6%) | 🟡 Medium |
| 9 | ONNX / INT8 / TensorRT / Jetson not started | 🟢 Later phase |
| 10 | Local `phase1_preprocessing.py` / `phase1_dataloader.py` don't match v0.2.0 | 🟢 Housekeeping |
