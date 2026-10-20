"""Genome and methylation tables -> per-cytosine label arrays, and training/validation windows."""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE = np.full(256, 4, np.uint8)
for _i, _b in enumerate('ACGT'):
    CODE[ord(_b)] = CODE[ord(_b.lower())] = _i
CONTEXTS = ('CG', 'CHG', 'CHH')  # codes 1, 2, 3; 0 = not a scored cytosine
FORMATS = ('cx', 'allc', 'bedmethyl')


def _open(path):
    return gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)


def read_fasta(path, names=None):
    """{name: uint8 codes A0 C1 G2 T3 N4}; the name is the first word of the header line."""
    seqs, name, buf = {}, None, []

    def flush():
        if name is not None and (names is None or name in names):
            seqs[name] = CODE[np.frombuffer(''.join(buf).encode(), np.uint8)]

    with _open(path) as f:
        for line in f:
            if line.startswith('>'):
                flush()
                name, buf = line[1:].split()[0], []
            elif names is None or name in names:
                buf.append(line.strip())
    flush()
    missing = set(names or ()) - set(seqs)
    if missing:
        raise ValueError(f'not in {path}: {sorted(missing)}')
    return seqs


def contexts(seq):
    """(2, L) uint8 context codes of plus-strand C (row 0) and minus-strand C, i.e. reference G (row 1)."""
    out = np.zeros((2, len(seq)), np.uint8)
    nxt1, nxt2 = np.r_[seq[1:], 4], np.r_[seq[2:], 4, 4]
    prv1, prv2 = np.r_[4, seq[:-1]], np.r_[4, 4, seq[:-2]]
    for h, base, b1, b2, g in ((0, 1, nxt1, nxt2, 2), (1, 2, prv1, prv2, 1)):
        m = (seq == base) & (b1 < 4) & (b2 < 4)
        out[h, m & (b1 == g)] = 1
        out[h, m & (b1 != g) & (b2 == g)] = 2
        out[h, m & (b1 != g) & (b2 != g)] = 3
    return out


def read_table(path, fmt, chunksize=2_000_000):
    """Yield chunks with columns chrom, pos (0-based), minus, beta, cov, ctx (1-3, or 0 if the format has none).
    cx:        Bismark CX report: chrom, pos (1-based), strand, n_methylated, n_unmethylated, context, ...
    allc:      methylpy allc: chrom, pos (1-based), strand, trinucleotide, mc, total, ...
    bedmethyl: chrom, start (0-based), end, name, score, strand, ..., coverage (col 10), percent (col 11)"""
    cols, text = {'cx': ([0, 1, 2, 3, 4, 5], (0, 2, 5)), 'allc': ([0, 1, 2, 3, 4, 5], (0, 2, 3)),
                  'bedmethyl': ([0, 1, 3, 5, 9, 10], (0, 3, 5))}[fmt]
    with _open(path) as f:
        first = f.readline().split()
    header = len(first) < 2 or not first[1].isdigit()
    for d in pd.read_csv(path, sep=r'\s+', header=None, usecols=cols, skiprows=int(header),
                         dtype={i: str for i in text}, chunksize=chunksize):
        if fmt == 'cx':
            cov = d[3] + d[4]
            beta, pos, strand = d[3] / cov.clip(lower=1), d[1] - 1, d[2]
            ctx = d[5].map({'CG': 1, 'CHG': 2, 'CHH': 3}).fillna(0)
        elif fmt == 'allc':
            cov = d[5]
            beta, pos, strand = d[4] / cov.clip(lower=1), d[1] - 1, d[2]
            tri = d[3].str.upper()
            ctx = np.where(tri.str[1] == 'G', 1, np.where(tri.str[2] == 'G', 2, 3))
        else:
            d = d[d[3] != 'h']  # modkit: keep 5mC rows, drop 5hmC
            cov, beta, pos, strand, ctx = d[9], d[10] / 100, d[1], d[5], np.zeros(len(d))
        yield pd.DataFrame(dict(chrom=d[0], pos=pos.astype(np.int64), minus=(strand == '-').astype(np.int8),
                                beta=beta.astype(np.float32), cov=cov.astype(np.int64),
                                ctx=np.asarray(ctx, np.uint8)))


def build_labels(fasta, table, fmt, chroms, out, min_cov, cap):
    """Write <chrom>.{seq,target,context,weight}.npy. A cytosine enters the loss if its reference context agrees
    with the table and its coverage is >= min_cov; its loss weight is min(coverage, cap) / cap."""
    out.mkdir(parents=True, exist_ok=True)
    seqs = read_fasta(fasta, set(chroms))
    arrays = {}
    for c in chroms:
        L = len(seqs[c])
        np.save(out / f'{c}.seq.npy', seqs[c])
        mm = lambda k, dt: np.lib.format.open_memmap(out / f'{c}.{k}.npy', mode='w+', dtype=dt, shape=(2, L))
        arrays[c] = (contexts(seqs[c]), mm('target', np.float32), mm('context', np.uint8), mm('weight', np.float16))
    n_rows = n_agree = 0
    for d in read_table(table, fmt):
        for c, g in d[d.chrom.isin(list(arrays))].groupby('chrom', sort=False):
            ref, target, context, weight = arrays[c]
            g = g[(g.pos >= 0) & (g.pos < ref.shape[1])]
            h, p, cov = g.minus.to_numpy(), g.pos.to_numpy(), g['cov'].to_numpy()
            rc, fc = ref[h, p], g.ctx.to_numpy()
            agree = (rc > 0) & ((fc == 0) | (fc == rc))
            n_rows += len(g)
            n_agree += int(agree.sum())
            k = agree & (cov >= min_cov)
            target[h[k], p[k]] = g.beta.to_numpy()[k]
            context[h[k], p[k]] = rc[k]
            weight[h[k], p[k]] = (np.minimum(cov[k], cap).astype(np.float32) / cap).astype(np.float16)
    if n_rows == 0 or n_agree < 0.95 * n_rows:
        raise ValueError(f'{n_agree} of {n_rows} table rows on {chroms} match a reference cytosine of the stated '
                         'strand and context: check chromosome names, assembly and coordinate base')
    counts = {c: {name: int((a[2] == i).sum()) for i, name in enumerate(CONTEXTS, 1)} for c, a in arrays.items()}
    for a in arrays.values():
        for m in a[1:]:
            m.flush()
    return dict(rows=n_rows, rows_agreeing=n_agree, sites=counts)


def prepare_labels(fasta, table, fmt, chroms, out, min_cov, cap):
    """Build the label arrays once (reused when the inputs are unchanged) and open them read-only."""
    out = Path(out)
    inputs = dict(fasta=str(Path(fasta).resolve()), table=str(Path(table).resolve()), format=fmt,
                  chroms=sorted(chroms), min_cov=min_cov, cap=cap)
    info = out / 'labels.json'
    if not info.exists() or json.loads(info.read_text())['inputs'] != inputs:
        summary = build_labels(fasta, table, fmt, chroms, out, min_cov, cap)
        info.write_text(json.dumps(dict(inputs=inputs, **summary), indent=1))
    keys = ('seq', 'target', 'context', 'weight')
    return {c: {k: np.load(out / f'{c}.{k}.npy', mmap_mode='r') for k in keys} for c in chroms}


def get_batch(labels, chroms, centers, window, core, rc=None):
    """Windows of `window` bp centred on `centers`; labels for the central `core` bp. rc: reverse-complement mask."""
    codes = np.stack([labels[c]['seq'][s - window // 2:s + window // 2] for c, s in zip(chroms, centers)])
    lab = lambda k: np.stack([labels[c][k][:, s - core // 2:s + core // 2] for c, s in zip(chroms, centers)])
    y, ctx, w = lab('target'), lab('context'), lab('weight').astype(np.float32)
    if rc is not None and rc.any():
        codes[rc] = np.where(codes[rc, ::-1] < 4, 3 - codes[rc, ::-1], 4)
        y[rc], ctx[rc], w[rc] = y[rc, ::-1, ::-1], ctx[rc, ::-1, ::-1], w[rc, ::-1, ::-1]
    return tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in (codes.astype(np.int64), y, ctx, w))


def sample_windows(rng, labels, chroms, n, margin):
    """Window centres uniform over the training chromosomes (by length); half of them reverse-complemented."""
    L = np.array([len(labels[c]['seq']) for c in chroms])
    pick = rng.choice(len(chroms), n, p=L / L.sum())
    centers = [int(rng.integers(margin, L[i] - margin)) for i in pick]
    return [chroms[i] for i in pick], centers, rng.random(n) < .5


def val_anchors(labels, chroms, n, core, margin, seed):
    """A fixed random set of n non-overlapping core blocks on the validation chromosomes."""
    pairs = [(c, s) for c in chroms for s in range(margin, len(labels[c]['seq']) - margin, core)]
    return [pairs[i] for i in np.random.default_rng(seed).permutation(len(pairs))[:n]]
