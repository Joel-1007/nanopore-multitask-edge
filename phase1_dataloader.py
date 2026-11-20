#!/usr/bin/env python3
"""
phase1_dataloader.py
=====================
PyTorch DataLoader + Smoke Test for Phase 1 WebDataset Shards
Edge-Deployable Multi-Task Nanopore Framework

Usage
-----
python phase1_dataloader.py --shards shards/nomiss/train --smoke
python phase1_dataloader.py --shards shards/nomiss/train --benchmark --num-workers 4

Dependencies
------------
pip install torch webdataset numpy tqdm
"""

import argparse
import glob
import io
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
import webdataset as wds
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

WINDOW_SIZE = 4096


# ─────────────────────────────────────────────
# Decoders
# ─────────────────────────────────────────────

def nanopore_decoder(key: str, data: bytes):
    if key.endswith('.npy'):
        return np.load(io.BytesIO(data), allow_pickle=False)
    if key.endswith('.json'):
        return json.loads(data.decode('utf-8'))
    return data


# ─────────────────────────────────────────────
# Collate: pads variable-length base/mod labels
# ─────────────────────────────────────────────

def collate_nanopore(batch: List[dict]) -> Dict[str, torch.Tensor]:
    """
    Returns
    -------
    signal        : (B, 1, 4096)   float32  — channel dim for 1D-Conv
    bases         : (B, L_max)     int64    — CTC targets, -1 = padding
    mods          : (B, L_max, 2)  float32  — [5mC, 6mA] probs, 0 = padding
    label_lengths : (B,)           int64    — valid label lengths for CTC loss
    mask          : (B, L_max)     bool     — True = valid position
    mod_positive  : (B,)           bool
    """
    B      = len(batch)
    lengths = [len(s['bases.npy']) for s in batch]
    L_max   = max(lengths)

    signal_t = torch.zeros(B, 1, WINDOW_SIZE)
    bases_t  = torch.full((B, L_max), -1, dtype=torch.long)
    mods_t   = torch.zeros(B, L_max, 2)
    mask_t   = torch.zeros(B, L_max, dtype=torch.bool)
    mod_pos  = []

    for i, (s, L) in enumerate(zip(batch, lengths)):
        signal_t[i, 0] = torch.from_numpy(s['signal.npy'])
        bases_t[i, :L] = torch.from_numpy(s['bases.npy'])
        mods_t[i, :L]  = torch.from_numpy(s['mods.npy'])
        mask_t[i, :L]  = True
        mod_pos.append(bool(s['meta.json'].get('mod_positive', False)))

    return {
        'signal':        signal_t.float(),
        'bases':         bases_t,
        'mods':          mods_t.float(),
        'label_lengths': torch.tensor(lengths, dtype=torch.long),
        'mask':          mask_t,
        'mod_positive':  torch.tensor(mod_pos, dtype=torch.bool),
    }


# ─────────────────────────────────────────────
# DataLoader builder
# ─────────────────────────────────────────────

def build_dataloader(
    shard_dir:   str,
    split:       str  = 'train',
    batch_size:  int  = 32,
    num_workers: int  = 2,
    shuffle:     bool = True,
) -> DataLoader:
    shards = sorted(glob.glob(str(Path(shard_dir) / f"{split}_*.tar")))
    if not shards:
        raise FileNotFoundError(f"No shards found: {Path(shard_dir) / f'{split}_*.tar'}")
    log.info(f"Found {len(shards)} shard(s) for split='{split}'.")

    dataset = (
        wds.WebDataset(shards, shardshuffle=shuffle)
        .decode(nanopore_decoder)
        .to_tuple('signal.npy', 'bases.npy', 'mods.npy', 'meta.json')
        .map(lambda s, b, m, meta: {
            'signal.npy': s, 'bases.npy': b,
            'mods.npy': m,  'meta.json': meta
        })
        .batched(batch_size, collation_fn=collate_nanopore, partial=False)
    )

    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ─────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────

def smoke_test(shard_dir: str, split: str, batch_size: int = 8, n_batches: int = 5):
    log.info("=" * 60)
    log.info(f"SMOKE TEST — split={split}  {n_batches} batches × {batch_size}")
    log.info("=" * 60)

    loader = build_dataloader(shard_dir, split=split, batch_size=batch_size,
                              num_workers=0, shuffle=False)

    signal_means, signal_stds, seq_lens, mod_rates = [], [], [], []
    total = 0

    for idx, batch in enumerate(loader):
        if idx >= n_batches:
            break

        sig  = batch['signal']           # (B, 1, 4096)
        base = batch['bases']            # (B, L_max)
        mod  = batch['mods']             # (B, L_max, 2)
        lens = batch['label_lengths']    # (B,)
        mask = batch['mask']             # (B, L_max)
        mpos = batch['mod_positive']     # (B,)
        B    = sig.shape[0]
        total += B

        # ── Shape assertions ──────────────────────────────────────────
        assert sig.shape == (B, 1, WINDOW_SIZE), f"Bad signal shape: {sig.shape}"
        assert mod.shape[-1] == 2,               f"Mods last dim must be 2"
        assert mask.shape == base.shape,         f"Mask/base shape mismatch"

        # ── Value assertions ──────────────────────────────────────────
        sig_flat    = sig.squeeze(1)
        means       = sig_flat.mean(dim=1)
        stds        = sig_flat.std(dim=1)
        valid_bases = base[mask]

        assert (means.abs() < 5.0).all(), f"Signal mean out of range: {means}"
        assert (stds > 0).all(),          f"Zero-variance signal"
        assert (valid_bases >= 0).all() and (valid_bases <= 4).all(), \
            f"Base labels out of range: {valid_bases.unique()}"
        assert (mod[mask] >= 0).all() and (mod[mask] <= 1).all(), \
            f"Mod probs out of [0,1]"

        signal_means.extend(means.tolist())
        signal_stds.extend(stds.tolist())
        seq_lens.extend(lens.tolist())
        mod_rates.append(mpos.float().mean().item())

        if idx == 0:
            log.info(f"\n[Batch 0 — shapes]")
            log.info(f"  signal        : {tuple(sig.shape)}  {sig.dtype}")
            log.info(f"  bases         : {tuple(base.shape)} {base.dtype}")
            log.info(f"  mods          : {tuple(mod.shape)}  {mod.dtype}")
            log.info(f"  label_lengths : min={lens.min()}  max={lens.max()}")
            log.info(f"  mod_positive  : {mpos.sum().item()} / {B} windows")

    log.info(f"\n[Aggregate — {total} windows]")
    log.info(f"  Signal mean : {np.mean(signal_means):.4f} ± {np.std(signal_means):.4f}  (expect ≈ 0)")
    log.info(f"  Signal std  : {np.mean(signal_stds):.4f} ± {np.std(signal_stds):.4f}   (expect ≈ 1)")
    log.info(f"  Seq length  : {np.mean(seq_lens):.1f} ± {np.std(seq_lens):.1f} bases/window")
    log.info(f"  Mod-pos rate: {np.mean(mod_rates)*100:.1f}%  (expect 20–80% for bacterial data)")
    log.info("\n✅ SMOKE TEST PASSED")


# ─────────────────────────────────────────────
# Throughput benchmark
# ─────────────────────────────────────────────

def benchmark(shard_dir: str, split: str,
              batch_size: int = 64, num_workers: int = 4, n_batches: int = 100):
    log.info("=" * 60)
    log.info(f"THROUGHPUT BENCHMARK — {n_batches} batches × {batch_size}")
    log.info("=" * 60)

    loader = build_dataloader(shard_dir, split=split, batch_size=batch_size,
                              num_workers=num_workers, shuffle=True)
    total   = 0
    t_start = time.perf_counter()

    for idx, batch in enumerate(tqdm(loader, total=n_batches, unit='batch')):
        total += batch['signal'].shape[0]
        if idx + 1 >= n_batches:
            break

    elapsed    = time.perf_counter() - t_start
    throughput = total / elapsed

    log.info(f"\nThroughput : {throughput:,.0f} windows/sec")
    log.info(f"Per-batch  : {elapsed / n_batches * 1000:.1f} ms")

    # RTX 3070 target: model forward ≈ 100–300 windows/sec in early training
    # DataLoader must sustain ≥ 500 windows/sec to stay ahead of GPU
    if throughput < 300:
        log.warning("⚠️  Low throughput — increase --num-workers or shard count.")
    else:
        log.info("✅ DataLoader will not bottleneck GPU training.")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Phase 1 DataLoader + Smoke Test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--shards',      required=True)
    p.add_argument('--split',       default='train', choices=['train', 'val', 'test'])
    p.add_argument('--batch-size',  type=int, default=32)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--smoke',       action='store_true')
    p.add_argument('--benchmark',   action='store_true')
    p.add_argument('--n-batches',   type=int, default=5)
    args = p.parse_args()

    if args.smoke:
        smoke_test(args.shards, args.split, args.batch_size, args.n_batches)
    if args.benchmark:
        benchmark(args.shards, args.split, args.batch_size, args.num_workers, args.n_batches * 20)
    if not args.smoke and not args.benchmark:
        p.print_help()


if __name__ == '__main__':
    main()
