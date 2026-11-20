# Project Implementation Summary: Edge-Deployable Multi-Task Nanopore Deep Learning Framework

> **Last Updated:** 2026-10-12 | **Source of truth:** `JADA FULL RESULTS.pdf` (52 pages, updated 12 Oct 2026)
>
> **Status:** Data preparation complete for the reduced scope (NO-MISS 0–2 + EF26). A compact dual-head student has been trained, evaluated against Dorado on NO-MISS validation windows, and exported to FP16. **New training is frozen for the review.** External (EF26) evaluation, full-read inference, INT8/TensorRT and Jetson deployment have not been done.

## 1. Project Identity & Overview
* **Project Title:** An Edge-Deployable Multi-Task Deep Learning Framework for Simultaneous Basecalling, Epigenetic Modification Detection, and Transcript Quantification from Nanopore Signals
* **Authors / Team:** Aditya, Joel, Dinesh & Ansh (Amrita Vishwa Vidyapeetham, Coimbatore) — Group 13, DNA Sequencing Technologies
* **Core Problem:** ONT pipelines (Dorado → alignment → quantification) are multi-stage and need datacenter-class GPUs; they cannot run in real time on low-power edge hardware.
* **Approach as implemented:** A separately trained compact CNN–Transformer **student** that reads normalized raw `.pod5` signal windows and outputs (A) a base sequence via CTC and (B) per-base 6mA / 4mC / 5mC probabilities. Dorado SUP v5.2.0 is the **teacher**: its decoded sequences and soft modification probabilities are the training labels (pseudo-labels, not biological ground truth).
* **Original deployment goal (not yet started):** INT8 TensorRT engine on an NVIDIA Jetson Orin Nano (~5.2 GB usable memory) with <5% relative accuracy drop versus the workstation model.

---

## 2. Team Hardware & Workload Allocation

| Device | VRAM / Memory | Role in Project | Notes |
| :--- | :--- | :--- | :--- |
| **RTX 3070 Laptop** | 8 GB VRAM | Dorado basecalling, all student training, evaluation and benchmarks | Every reported training run and timing used this GPU (PyTorch 2.7.1+cu126). |
| **RTX 4050 Laptop** | 6 GB VRAM | Secondary training (planned sweeps) | Not used in reported results. |
| **RTX 3060 Laptop** | 6 GB VRAM (60W cap) | Secondary training (planned sweeps) | Not used in reported results. |
| **M2 Pro Mac** | 16 GB Unified Memory | Report writing, documentation, code review | Cannot run TensorRT (Phase 3/4). |

---

## 3. Scope Decisions (current)

1. **Reduced training scope — NO-MISS files 0–2 only.** All eight NO-MISS POD5 files (0–7) were downloaded, but files **3–7 are deliberately excluded** from the initial experiment. They are not pending work.
2. **Independent test set — EF26.** The local `prjna_test.pod5` is linked by read ID to `SRR32250932.fast5`, which corresponds to *Enterococcus faecium* isolate **EF26** in the published accession table (Scientific Data 2025, https://doi.org/10.1038/s41597-025-06319-4). EF26 is reserved **only** for external evaluation and was never used for training or checkpoint selection. (The original plan cited `SRR28777746`, *Listeria*; the team intentionally switched on 2026-08-03.)
3. **Circular-label caveat.** All labels come from Dorado. Every metric reported so far is **agreement with Dorado**, not biological accuracy. Independent references and modification controls for EF26 are still required.
4. **Head C (transcript quantification).** Not implemented. Both datasets are genomic DNA, so the implemented student is **dual-head** (basecalling + modification). Transcript quantification stays as future work.

---

## 4. Datasets — Final State

| Dataset | Role | BAM records | Tags (MM/ML/mv/ts/ns) | Windows | Parents | Shards |
|---|---|---|---|---|---|---|
| NO-MISS 0 | Train/val/test | 309,432 | 100% | 2,048,292 | 304,551 | — |
| NO-MISS 1 | Train/val/test | 286,315 | 100% | 2,048,181 | 280,782 | — |
| NO-MISS 2 | Train/val/test | 272,684 | 100% | 2,052,140 | 266,984 | 51 |
| **NO-MISS 0–2 total** | | 868,431 | 100% | **6,148,613** | **852,317** | **153** |
| **EF26 (SRR32250932)** | External test only | 124,016 | 100% | **2,155,099** | **123,514** | **51** |

* NO-MISS splits: SHA256(`42:parent_id`), 70/15/15 at **parent-read** level (split children and overlapping windows stay together). Combined training split = **4,310,089 windows**. Splits are not modification-stratified or species-disjoint, and the internal test split shares a sequencing run with training.
* NO-MISS 0–2 are partitions of one run (`853a956f…`, flow cell PBK98658), not biological replicates.
* EF26 was re-basecalled with move tables (Dorado 2.1.2) on 4 Oct 2026. The earlier blocker ("PRJNA BAM has no move tables") is **resolved**.
* Preprocessing v0.2.0: median–MAD normalization per called segment, 4,096-sample windows with 3,072 hop (25% overlap), short windows zero-padded with validity mask, max parent length 2,000,000 samples. Dataset dirs: `datasets/nomiss2_full_seed42_v020`, `datasets/ef26_external_full_v021`.

---

## 5. Model & Results (validation against Dorado)

### Student architecture
Two kernel-5 stride-2 convolutions (total stride 4) → sinusoidal positions → 2-layer pre-norm Transformer (4 heads) → base head (5 classes: blank, A, C, G, T; CTC) and modification head (6mA, 4mC, 5mC; masked soft-target BCE). Loss = CTC + 1.0 × BCE.

| Variant | Width / FF | Parameters | Steps trained | Evaluated checkpoint |
|---|---|---|---|---|
| Baseline | 96 / 288 | 234,728 | 50,000 | step 49,000 |
| **Selected** | **192 / 576** | **930,248** | **100,000** | **step 96,000** |

Training: AdamW lr 3e-4, wd 0.01, microbatch 4 × accumulation 4 (effective 16), AMP, grad clip 1.0, seed 42. 100K steps ≈ 1.6M windows ≈ 37% of one epoch.

### Headline numbers (4,096 NO-MISS validation windows, seed 123)

| Metric | Width 96 | **Width 192** |
|---|---|---|
| Window CER vs Dorado | 29.42% | **21.73%** |
| 6mA AP / recall@0.5 | 0.139 / 0.25% | 0.264 / 9.67% |
| 4mC AP / recall@0.5 | 0.009 / 0% | 0.011 / **0% (no positive calls)** |
| 5mC AP / recall@0.5 | 0.264 / 7.99% | **0.651 / 54.74%** |

* **FP16 export:** 4.52 MB → 2.27 MB (−49.86%, 1.994×). CER 21.7291% vs 21.7304%, so quality is unchanged. This is the deployment candidate.
* **Decoder:** beam width 10 gives 20.26% CER vs 21.35% greedy (256 windows), but takes 93 ms vs 0.1 ms CPU per window. Greedy is kept.
* **Forward latency (RTX 3070, batch 1, warm cache, model only):** ~2.15–2.20 ms per window. This excludes I/O, normalization, decoding and stitching.
* **Dorado 100-read pilot:** 5.88 s wall time (1.60 s logged processing). The paired student run is **not yet done**, so there is no speedup claim.

---

## 6. Roadmap Status

| Phase | Plan | Status (12 Oct 2026) |
|---|---|---|
| 1 — Data engineering | Dorado labels, normalization, windowing, shards | ✅ **Complete** for NO-MISS 0–2 + EF26 |
| 2 — Workstation training | Shared encoder + CTC head + mod head | ✅ Functional student trained and evaluated vs Dorado. ⚠️ Accuracy is insufficient for deployment (21.7% CER, 4mC fails). Training frozen for review. |
| 3 — Compression | ONNX → TensorRT INT8 PTQ, FP16 fallback | 🟡 Only FP16 weight storage done. ONNX, INT8 and pruning not started. |
| 4 — Edge benchmarking | Jetson Orin Nano throughput, power, accuracy | ⬜ Not started |

## 7. Open Work (priority order)

1. **External evaluation on EF26.** Add an explicit external-evaluation option (the current eval command rejects a manifest that differs from the training manifest). First verify EF26 ↔ NO-MISS read-ID disjointness.
2. **Independent references.** Obtain a verified EF26 reference genome and independent modification annotations/controls; align both Dorado and student output with the same protocol.
3. **Full-read inference.** Window stitching plus modification-coordinate reconstruction.
4. **Train/inference alignment gap.** Modification logits are trained at Dorado move positions but read at student-decoded positions during inference.
5. **Paired Dorado vs student benchmark** on the same 100 reads, with a stated, comparable scope. Record GPU memory and power.
6. **Model quality:** 4mC detection (positive weighting / focal loss with a matched AP diagnostic), longer training, and deletion-dominated sequence errors (50.6% of edits).
7. **Phase 3/4:** ONNX export, INT8 PTQ, Jetson deployment.

> **Note on local code:** `phase1_preprocessing.py` and `phase1_dataloader.py` in this folder do **not** match preprocessing v0.2.0 as described in the results (e.g. they use a Python `hash()` split, two mod channels and no padding mask). Treat them as early drafts, not the code that produced the datasets.
