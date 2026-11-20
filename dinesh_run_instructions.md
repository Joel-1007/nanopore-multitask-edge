# Instructions for Dinesh — Current State and Next Steps
**Machine:** `dines@dines:~/nanopore_project`  
**Updated:** 2026-10-12 (matches `JADA FULL RESULTS.pdf`)  
**Context:** Data preparation and student training are done. **Do not start new training runs. Training is frozen for the review.**

---

## Status of the old stages (A–F from the 27 Sep version)

| Stage | Old task | Status |
|---|---|---|
| A | PRJNA tag audit, split counts, sort NO-MISS 0 | ✅ Done. EF26 re-basecalled with move tables; all 124,016 records carry MM/ML/mv/ts/ns |
| B | Basecall NO-MISS 1–7 | ✅ Files 1–2 done. ⛔ **Do not run files 3–7.** They are out of scope |
| C | Verify all 9 BAMs | ✅ Replaced by per-file audits of NO-MISS 0–2 + EF26 |
| D | Install preprocessing deps | ✅ Done (Python 3.12.3, NumPy 2.5.0, pysam 0.24.0, POD5 0.3.39) |
| E | Preprocess (merged BAM) | ✅ Done per file, no merge: NO-MISS 0–2 = 6,148,613 windows; EF26 = 2,155,099 test-only windows |
| F | DataLoader smoke test | ✅ Superseded. Student training ran 100K steps on these shards |

> The old `samtools merge` of all 8 BAMs and the `phase1_preprocessing.py` commands from the previous version are **obsolete**. Do not run them.

---

## Before anything
```bash
cd ~/nanopore_project
source venv/bin/activate
```

---

## STEP 1 — Preserve the evidence (do first, ~10 min)

Don't delete or overwrite any of these:

| Item | Path |
|---|---|
| NO-MISS BAMs | `bam_output/nomiss_0_resumed_20260923_213852.bam`, NO-MISS 1 BAM, `bam_output/nomiss_2_moves_v212_20260929T105046Z.bam` |
| EF26 BAM | `bam_output/ef26_moves_v212_b32_m128_20261004T173851_496171996Z.bam` |
| Datasets | `datasets/nomiss2_full_seed42_v020`, `datasets/ef26_external_full_v021` (plus NO-MISS 0 and 1 dataset dirs) |
| W96 run | `runs/student_baseline_pilot_20261008T075006_726080466Z` |
| W192 run | `student_w192_20261009T131226_647430928Z` (evaluated checkpoint = step 96,000) |
| FP16 export | `results/student_w192_precision_20261012T092355_798481217Z/student_fp16_weights.pt` |
| 100-read pilot | `results/paired_100_reads_20261012T094150_770570Z` |

Record hashes so we can prove the files haven't changed:
```bash
mkdir -p logs
find runs results -name '*.pt' -print0 | xargs -0 sha256sum > logs/checkpoint_hashes_$(date +%Y%m%d).txt
cat logs/checkpoint_hashes_$(date +%Y%m%d).txt
```
> **Expected:** the FP16 export hash starts with `bcec4cf7…` and the step-96,000 source checkpoint hash starts with `b02b5c7e…`.

---

## STEP 2 — EF26 ↔ NO-MISS read-ID disjointness check (~10 min)

This hasn't been verified yet and is required before we evaluate on EF26.

```bash
mkdir -p logs/disjoint
samtools view bam_output/ef26_moves_v212_b32_m128_20261004T173851_496171996Z.bam \
  | cut -f1 | sort -u > logs/disjoint/ef26_ids.txt

for B in bam_output/nomiss_0_resumed_20260923_213852.bam \
         <NO-MISS 1 BAM path> \
         bam_output/nomiss_2_moves_v212_20260929T105046Z.bam; do
  samtools view "$B" | cut -f1
done | sort -u > logs/disjoint/nomiss_ids.txt

wc -l logs/disjoint/*.txt
comm -12 logs/disjoint/ef26_ids.txt logs/disjoint/nomiss_ids.txt | wc -l
```
> **Expected:** `ef26_ids.txt` has 124,016 lines, and the overlap count is **0**. If it isn't 0, stop and message Joel.

---

## STEP 3 — Student side of the 100-read paired benchmark

Dorado's half is already done (`results/paired_100_reads_20261012T094150_770570Z`: 5.88 s wall, 1.60 s logged). Joel will send the student inference command for the same `input_100.pod5`. When you run it:
- Use the step-96,000 FP16 weights, batch size as specified, and greedy decoding.
- Time the **whole process** (start → output written), same as the Dorado wall-time measurement.
- In a second terminal, log GPU memory and power while it runs:
  ```bash
  nvidia-smi --query-gpu=timestamp,memory.used,power.draw --format=csv -lms 200 > logs/paired_100_gpu.csv
  ```
- Then do the same for Dorado on `input_100.pod5`, so both sides have GPU traces.

---

## STEP 4 — EF26 external evaluation (after Steps 2–3)

Blocked until the evaluation script has an explicit external-dataset option (the current one rejects a manifest that differs from the training one). Joel will send the updated command. Use `datasets/ef26_external_full_v021`, checkpoint step 96,000, and the same metrics as the validation report (CER, per-channel AP / precision / recall).

---

## What to message Joel

| Step | What to send |
|---|---|
| 1 | `logs/checkpoint_hashes_*.txt` |
| 2 | Output of `wc -l` and the overlap count |
| 3 | Student timing JSON + `logs/paired_100_gpu.csv` for both runs |
| 4 | Evaluation JSON |

---

## Disk usage (for reference)
NO-MISS 3–7 POD5s (~5 GB each) are no longer needed for this experiment. **Ask Joel before deleting them.** We may still use them later.
