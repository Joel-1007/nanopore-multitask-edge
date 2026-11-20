#!/usr/bin/env python3
"""
phase1_preprocessing.py
========================
Phase 1 Signal Preprocessing Pipeline
Edge-Deployable Multi-Task Nanopore Framework
Amrita Vishwa Vidyapeetham — Team: Aditya, Joel, Dinesh, Ansh

Workflow
--------
BAM (Dorado MM/ML/mv/ts/ns tags)
         +
POD5 (raw ionic current signal)
         |
         v
  ┌──────────────────────────────┐
  │ 1. Read ID cross-reference   │
  │ 2. Signal extraction (pA)    │
  │ 3. Z-score/MAD normalization │
  │ 4. 4096-sample windowing     │
  │    (25 % overlap = 3072 step)│
  │ 5. Move-table base mapping   │
  │ 6. MM/ML mod label slicing   │
  │ 7. Stratified shard export   │
  └──────────────────────────────┘
         |
         v
  WebDataset .tar shards
    {key}.signal.npy   – (4096,)     float32  normalized signal
    {key}.bases.npy    – (L,)        int64    CTC target  [A=0 C=1 G=2 T=3]
    {key}.mods.npy     – (L, 2)      float32  [5mC_prob, 6mA_prob]  in [0,1]
    {key}.meta.json    – read_id, organism, split, window_idx, mod_positive

Usage
-----
# NO-MISS training set (all 8 files, sorted+merged BAM)
python phase1_preprocessing.py \\
    --bam   bam/nomiss/nomiss_training.bam \\
    --pod5  pod5/raw/ \\
    --out   shards/nomiss \\
    --split train \\
    --organism nomiss \\
    --shard-size 2000

# PRJNA independent test set
python phase1_preprocessing.py \\
    --bam   bam/prjna/prjna_test.bam \\
    --pod5  pod5/raw/prjna_test.pod5 \\
    --out   shards/prjna \\
    --split test \\
    --organism prjna_efaecium \\
    --shard-size 1000

Dependencies
------------
pip install pysam pod5 numpy webdataset tqdm
"""

import argparse
import json
import logging
import os
import random
import sys
import tarfile
import io
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pysam
import pod5
import webdataset as wds
from tqdm import tqdm

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

WINDOW_SIZE   = 4096          # raw signal samples per window
OVERLAP       = 0.25          # 25% overlap → step = 3072 samples
WINDOW_STEP   = int(WINDOW_SIZE * (1.0 - OVERLAP))  # 3072

# Minimum number of bases in a window for it to be useful for CTC
MIN_BASES_PER_WINDOW = 10

# MAD scale factor to make it consistent with std-dev for Gaussian data
MAD_SCALE = 1.4826

# Base alphabet
BASE_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

# Modification tags Dorado writes for 4mC_5mC + 6mA models
# (C, 0, 'm') = 5mC  |  (C, 0, 'h') = 5hmC  |  (A, 0, 'a') = 6mA
MOD_CHANNELS = {
    '5mC': ('C', 0, 'm'),
    '6mA': ('A', 0, 'a'),
}

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Normalization
# ─────────────────────────────────────────────

def normalize_mad(signal: np.ndarray) -> np.ndarray:
    """
    Z-score using median and MAD (median absolute deviation).
    More robust to outlier spikes than mean/std.

    Formula:  z = (x - median) / (MAD * 1.4826)
    """
    median = np.median(signal)
    mad    = np.median(np.abs(signal - median))
    # Guard against flat signal (e.g., open-pore with no molecule)
    if mad < 1e-6:
        std = np.std(signal)
        scale = std if std > 1e-6 else 1.0
    else:
        scale = mad * MAD_SCALE
    return (signal - median) / scale


# ─────────────────────────────────────────────
# 2. Move-table signal-to-base mapping
# ─────────────────────────────────────────────

def parse_move_table(mv_tag) -> Tuple[int, np.ndarray]:
    """
    Parse the Dorado `mv:B:c` BAM tag.

    Format:
      mv_tag[0]  = stride  (raw samples per CTC frame, typically 5 for R10.4.1 400bps)
      mv_tag[1:] = move bits: 1 = new base starts at this CTC frame, 0 = continuation

    Returns
    -------
    stride : int
    moves  : np.ndarray of uint8, shape (n_ctc_frames,)
    """
    mv_array = np.frombuffer(bytes(mv_tag), dtype=np.uint8)
    stride   = int(mv_array[0])
    moves    = mv_array[1:].copy()
    return stride, moves


def build_frame_to_base(moves: np.ndarray) -> np.ndarray:
    """
    Build an array of shape (n_ctc_frames,) where each element is the
    0-indexed base position in the read sequence for that CTC frame.
    Frames before the first move get index -1 (invalid).

    The base index increments each time moves[i] == 1.
    """
    cumsum = np.cumsum(moves, dtype=np.int32)
    frame_to_base = cumsum - 1   # 0-indexed; -1 where no base has started yet
    return frame_to_base


def get_bases_in_window(
    w_start:       int,
    stride:        int,
    moves:         np.ndarray,
    frame_to_base: np.ndarray,
) -> np.ndarray:
    """
    Return the sorted array of unique base indices (into the read sequence)
    whose signal falls within the window [w_start : w_start + WINDOW_SIZE].

    A base 'falls in' the window if the CTC frame that introduced it (move=1)
    starts within the window's sample range.
    """
    # CTC frame range that overlaps this window
    frame_lo = w_start // stride
    frame_hi = min((w_start + WINDOW_SIZE - 1) // stride, len(moves) - 1)

    if frame_lo > frame_hi or frame_lo >= len(moves):
        return np.array([], dtype=np.int32)

    # Frames in this range where a new base starts (move == 1)
    local_moves = moves[frame_lo:frame_hi + 1]
    move_mask   = local_moves == 1
    if not np.any(move_mask):
        return np.array([], dtype=np.int32)

    local_indices = np.where(move_mask)[0] + frame_lo  # absolute frame indices
    base_indices  = frame_to_base[local_indices]

    # Filter out any invalid (negative) indices
    base_indices = base_indices[base_indices >= 0]
    return np.unique(base_indices).astype(np.int32)


# ─────────────────────────────────────────────
# 3. Modification label extraction (MM / ML)
# ─────────────────────────────────────────────

def parse_mod_labels(
    read: pysam.AlignedSegment,
    seq_len: int,
) -> np.ndarray:
    """
    Parse MM:Z and ML:B tags from a Dorado BAM record.

    Returns
    -------
    mod_probs : np.ndarray shape (seq_len, 2)  float32
        Column 0 = 5mC probability in [0, 1]
        Column 1 = 6mA  probability in [0, 1]
    """
    mod_probs = np.zeros((seq_len, 2), dtype=np.float32)

    try:
        # pysam.AlignedSegment.modified_bases returns:
        # Dict[(canonical_base_char, strand, mod_code_char)] -> List[(read_pos, quality_0_255)]
        mod_bases = read.modified_bases  # requires pysam >= 0.19
        if mod_bases is None:
            return mod_probs

        # 5mC
        key_5mc = MOD_CHANNELS['5mC']   # ('C', 0, 'm')
        if key_5mc in mod_bases:
            for read_pos, qual in mod_bases[key_5mc]:
                if 0 <= read_pos < seq_len:
                    mod_probs[read_pos, 0] = qual / 255.0

        # 6mA
        key_6ma = MOD_CHANNELS['6mA']   # ('A', 0, 'a')
        if key_6ma in mod_bases:
            for read_pos, qual in mod_bases[key_6ma]:
                if 0 <= read_pos < seq_len:
                    mod_probs[read_pos, 1] = qual / 255.0

    except (AttributeError, KeyError, ValueError):
        # Fallback: return zero labels (no modification info available)
        pass

    return mod_probs


# ─────────────────────────────────────────────
# 4. POD5 signal reader
# ─────────────────────────────────────────────

def build_pod5_index(pod5_paths: List[Path]) -> Dict[str, Path]:
    """
    Scan all POD5 files and build a dict: read_id (str) → pod5_path.
    Used to quickly look up which POD5 file contains a given read.
    """
    log.info(f"Indexing {len(pod5_paths)} POD5 file(s)…")
    index: Dict[str, Path] = {}
    for p in pod5_paths:
        with pod5.Reader(str(p)) as reader:
            for read in reader.reads():
                index[str(read.read_id)] = p
    log.info(f"Indexed {len(index):,} reads across {len(pod5_paths)} POD5 file(s).")
    return index


def get_signal_pa(read_id: str, pod5_index: Dict[str, Path]) -> Optional[np.ndarray]:
    """
    Extract the calibrated pA signal for a single read.
    Returns None if the read is not found.
    """
    if read_id not in pod5_index:
        return None
    pod5_path = pod5_index[read_id]
    with pod5.Reader(str(pod5_path)) as reader:
        for rec in reader.reads(selection=[read_id]):
            return rec.signal_pa.astype(np.float32)
    return None


# ─────────────────────────────────────────────
# 5. Window generator (core logic)
# ─────────────────────────────────────────────

def generate_windows(
    read:       pysam.AlignedSegment,
    pod5_index: Dict[str, Path],
    organism:   str,
) -> Generator[dict, None, None]:
    """
    For a single BAM record, yield one dict per valid signal window.

    Each dict contains:
      signal     : np.ndarray (WINDOW_SIZE,)   float32 — normalized signal
      bases      : np.ndarray (L,)              int64   — base indices [0-4]
      mods       : np.ndarray (L, 2)            float32 — [5mC_prob, 6mA_prob]
      read_id    : str
      organism   : str
      window_idx : int
      mod_positive : bool — True if any base in window has prob > 0.5
    """
    read_id = read.query_name
    seq     = read.query_sequence
    if seq is None or len(seq) == 0:
        return

    seq_len = len(seq)

    # ── 1. Get required tags ─────────────────────────────────────────────
    try:
        mv_raw = read.get_tag('mv')
        ts     = int(read.get_tag('ts'))
        ns     = int(read.get_tag('ns'))
    except KeyError:
        log.debug(f"  Skipping {read_id}: missing mv/ts/ns tags.")
        return

    # ── 2. Parse move table ─────────────────────────────────────────────
    stride, moves = parse_move_table(mv_raw)
    frame_to_base = build_frame_to_base(moves)

    # Sanity: number of bases implied by move table should match seq length
    n_bases_in_moves = int(np.sum(moves))
    if abs(n_bases_in_moves - seq_len) > 5:  # allow tiny discrepancy
        log.debug(f"  Skipping {read_id}: move-table base count {n_bases_in_moves} ≠ seq len {seq_len}")
        return

    # ── 3. Get signal ────────────────────────────────────────────────────
    signal_pa = get_signal_pa(read_id, pod5_index)
    if signal_pa is None:
        log.debug(f"  Skipping {read_id}: not found in POD5.")
        return

    # The read's signal in the POD5 spans [ts : ts+ns]
    read_signal = signal_pa[ts : ts + ns]
    if len(read_signal) < WINDOW_SIZE:
        return  # Read too short for even one window

    # ── 4. Normalize entire read signal ──────────────────────────────────
    read_signal = normalize_mad(read_signal)

    # ── 5. Parse modification labels ─────────────────────────────────────
    mod_probs = parse_mod_labels(read, seq_len)

    # ── 6. Slide windows ─────────────────────────────────────────────────
    window_idx = 0
    for w_start in range(0, len(read_signal) - WINDOW_SIZE + 1, WINDOW_STEP):
        signal_window = read_signal[w_start : w_start + WINDOW_SIZE].copy()

        # Map window to base indices
        base_indices = get_bases_in_window(w_start, stride, moves, frame_to_base)
        if len(base_indices) < MIN_BASES_PER_WINDOW:
            window_idx += 1
            continue

        # Extract base labels (CTC target)
        bases_in_seq = [BASE_TO_INT.get(seq[i], 4) for i in base_indices]
        bases_arr    = np.array(bases_in_seq, dtype=np.int64)

        # Extract modification probability labels
        mods_arr = mod_probs[base_indices].copy()  # shape (L, 2)

        mod_positive = bool(np.any(mods_arr > 0.5))

        yield {
            'signal':       signal_window.astype(np.float32),
            'bases':        bases_arr,
            'mods':         mods_arr,
            'read_id':      read_id,
            'organism':     organism,
            'window_idx':   window_idx,
            'mod_positive': mod_positive,
        }
        window_idx += 1


# ─────────────────────────────────────────────
# 6. Stratified split assignment
# ─────────────────────────────────────────────

def assign_split(
    read_id:   str,
    split:     str,
    val_frac:  float = 0.10,
    test_frac: float = 0.10,
    seed:      int   = 42,
) -> str:
    """
    Deterministically assign a read to train / val / test using
    a hash of the read_id. Stratified at the read level so windows
    from the same read never span multiple splits.

    If `split` is forced (e.g. 'test' for PRJNA), that is returned directly.
    """
    if split in ('val', 'test'):
        return split  # forced assignment (e.g. PRJNA always goes to test)

    rng = random.Random(hash(read_id + str(seed)))
    r   = rng.random()
    if r < val_frac:
        return 'val'
    elif r < val_frac + test_frac:
        return 'test'
    return 'train'


# ─────────────────────────────────────────────
# 7. WebDataset shard writer
# ─────────────────────────────────────────────

class ShardWriter:
    """
    Writes windows to WebDataset .tar shards.

    Shard layout:
      {split}_{shard_idx:06d}.tar
        └── {global_idx:010d}.signal.npy
        └── {global_idx:010d}.bases.npy
        └── {global_idx:010d}.mods.npy
        └── {global_idx:010d}.meta.json
    """

    def __init__(self, out_dir: Path, split: str, shard_size: int):
        self.out_dir    = out_dir
        self.split      = split
        self.shard_size = shard_size
        out_dir.mkdir(parents=True, exist_ok=True)

        self.shard_idx   = 0
        self.window_idx  = 0   # global window counter
        self.in_shard    = 0   # windows in current shard
        self._tarfile    = None
        self._open_shard()

        self.stats = {'total': 0, 'mod_positive': 0, 'mod_negative': 0}

    def _open_shard(self):
        if self._tarfile is not None:
            self._tarfile.close()
        shard_name = f"{self.split}_{self.shard_idx:06d}.tar"
        shard_path = self.out_dir / shard_name
        self._tarfile = tarfile.open(shard_path, 'w')
        log.info(f"  Opened shard: {shard_path}")
        self.in_shard = 0

    def _add_numpy(self, key: str, arr: np.ndarray):
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)
        info = tarfile.TarInfo(name=key)
        info.size = buf.getbuffer().nbytes
        self._tarfile.addfile(info, buf)

    def _add_json(self, key: str, obj: dict):
        data = json.dumps(obj).encode('utf-8')
        buf  = io.BytesIO(data)
        info = tarfile.TarInfo(name=key)
        info.size = len(data)
        self._tarfile.addfile(info, buf)

    def write(self, window: dict):
        prefix = f"{self.window_idx:010d}"

        self._add_numpy(f"{prefix}.signal.npy", window['signal'])
        self._add_numpy(f"{prefix}.bases.npy",  window['bases'])
        self._add_numpy(f"{prefix}.mods.npy",   window['mods'])
        self._add_json(f"{prefix}.meta.json", {
            'read_id':      window['read_id'],
            'organism':     window['organism'],
            'split':        self.split,
            'window_idx':   window['window_idx'],
            'mod_positive': window['mod_positive'],
        })

        self.window_idx += 1
        self.in_shard   += 1
        self.stats['total'] += 1
        if window['mod_positive']:
            self.stats['mod_positive'] += 1
        else:
            self.stats['mod_negative'] += 1

        if self.in_shard >= self.shard_size:
            self.shard_idx += 1
            self._open_shard()

    def close(self):
        if self._tarfile is not None:
            self._tarfile.close()
        log.info(
            f"[{self.split}] Wrote {self.stats['total']:,} windows | "
            f"mod+ {self.stats['mod_positive']:,} | "
            f"mod- {self.stats['mod_negative']:,} | "
            f"{self.shard_idx + 1} shards"
        )
        # Write a shard manifest
        manifest_path = self.out_dir / f"{self.split}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump({
                'split':         self.split,
                'n_windows':     self.stats['total'],
                'mod_positive':  self.stats['mod_positive'],
                'mod_negative':  self.stats['mod_negative'],
                'n_shards':      self.shard_idx + 1,
                'shard_size':    self.shard_size,
                'window_size':   WINDOW_SIZE,
                'window_step':   WINDOW_STEP,
            }, f, indent=2)
        log.info(f"  Manifest: {manifest_path}")


# ─────────────────────────────────────────────
# 8. Main orchestrator
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: Signal preprocessing → WebDataset shards',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--bam',        required=True,  help='Dorado BAM file (sorted, with MM/ML/mv/ts/ns tags)')
    parser.add_argument('--pod5',       required=True,  help='POD5 file or directory containing POD5 files')
    parser.add_argument('--out',        required=True,  help='Output directory for .tar shards')
    parser.add_argument('--split',      default='train', choices=['train', 'val', 'test'],
                        help='Force all reads to this split (use "train" for NO-MISS, "test" for PRJNA)')
    parser.add_argument('--organism',   default='unknown', help='Organism label written to meta.json')
    parser.add_argument('--shard-size', type=int, default=2000, help='Windows per .tar shard')
    parser.add_argument('--val-frac',   type=float, default=0.10, help='Fraction of reads for validation (train split only)')
    parser.add_argument('--test-frac',  type=float, default=0.00, help='Fraction of reads for test (train split only, usually 0 for NO-MISS)')
    parser.add_argument('--max-reads',  type=int, default=None, help='Limit number of BAM records processed (for testing)')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Phase 1 Preprocessing — Nanopore Multi-Task Framework")
    log.info("=" * 60)
    log.info(f"  BAM:      {args.bam}")
    log.info(f"  POD5:     {args.pod5}")
    log.info(f"  Output:   {args.out}")
    log.info(f"  Split:    {args.split}")
    log.info(f"  Organism: {args.organism}")
    log.info(f"  Shard sz: {args.shard_size}")
    log.info(f"  Window:   {WINDOW_SIZE} samples, {WINDOW_STEP} step ({int(OVERLAP*100)}% overlap)")

    # ── Collect POD5 paths ────────────────────────────────────────────
    pod5_input = Path(args.pod5)
    if pod5_input.is_dir():
        pod5_paths = sorted(pod5_input.glob('*.pod5'))
    else:
        pod5_paths = [pod5_input]

    if not pod5_paths:
        log.error(f"No POD5 files found at: {args.pod5}")
        sys.exit(1)

    # ── Build POD5 index ──────────────────────────────────────────────
    pod5_index = build_pod5_index(pod5_paths)

    # ── Open output shard writers (one per split) ─────────────────────
    out_dir = Path(args.out)
    writers: Dict[str, ShardWriter] = {}

    def get_writer(split_name: str) -> ShardWriter:
        if split_name not in writers:
            writers[split_name] = ShardWriter(out_dir / split_name, split_name, args.shard_size)
        return writers[split_name]

    # ── Process BAM ───────────────────────────────────────────────────
    bam = pysam.AlignmentFile(args.bam, 'rb', check_sq=False)
    n_reads     = 0
    n_windows   = 0
    n_skipped   = 0

    pbar = tqdm(unit='reads', desc='Processing')
    for read in bam.fetch(until_eof=True):
        if read.is_supplementary or read.is_secondary:
            continue  # process primary reads only

        if args.max_reads and n_reads >= args.max_reads:
            break

        n_reads += 1
        pbar.update(1)

        # Determine split for this read
        split_name = assign_split(
            read.query_name, args.split,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )

        got_any = False
        for window in generate_windows(read, pod5_index, args.organism):
            get_writer(split_name).write(window)
            n_windows += 1
            got_any = True

        if not got_any:
            n_skipped += 1

    pbar.close()
    bam.close()

    # ── Close writers ─────────────────────────────────────────────────
    for writer in writers.values():
        writer.close()

    log.info("─" * 60)
    log.info(f"Done. Reads processed: {n_reads:,} | Reads skipped: {n_skipped:,}")
    log.info(f"Total windows written: {n_windows:,}")
    log.info(f"Splits created: {list(writers.keys())}")
    log.info("─" * 60)


if __name__ == '__main__':
    main()
