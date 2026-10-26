"""Predict the methylation level of every cytosine (both strands) in a FASTA with a trained REED model.

python predict.py --checkpoint reed.pt --fasta genome.fa --region 4:1000001-1100000 --out pred.tsv.gz

Output (tab-separated, with header): chrom, pos (1-based), strand, context, p.
"""
import argparse
import gzip

import numpy as np
import pandas as pd
import torch

from reed.data import CONTEXTS, contexts, read_fasta
from reed.model import load_model


def predict_region(model, seq, start, end, window=10000, core=512, batch=16, device='cpu'):
    """(2, end - start) probabilities for seq[start:end]. Each block of `core` bp is read from the `window` bp
    centred on it; sequence beyond the chromosome ends is N."""
    L, mid = len(seq), (window - core) // 2
    out = np.empty((2, end - start), np.float32)
    centers = np.arange(start + core // 2, end + core // 2, core)
    for i in range(0, len(centers), batch):
        cc = centers[i:i + batch]
        codes = np.full((len(cc), window), 4, np.int64)
        for k, c in enumerate(cc):
            a = c - window // 2
            lo, hi = max(a, 0), min(a + window, L)
            codes[k, lo - a:hi - a] = seq[lo:hi]
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(codes).to(device)))[:, :, mid:mid + core].cpu().numpy()
        for k, c in enumerate(cc):
            a = c - core // 2 - start
            b = min(a + core, end - start)
            out[:, a:b] = p[k, :, :b - a]
    return out


def parse_regions(args):
    regions = []
    for r in args.region or []:
        chrom, _, span = r.partition(':')
        s, e = span.replace(',', '').split('-') if span else (1, None)
        regions.append((chrom, int(s) - 1, None if e is None else int(e)))
    if args.bed:
        for line in open(args.bed):
            f = line.split()
            if f and not line.startswith(('#', 'track', 'browser')):
                regions.append((f[0], int(f[1]), int(f[2])))
    return regions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--fasta', required=True)
    ap.add_argument('--region', action='append', help='chrom or chrom:start-end (1-based, inclusive); repeatable')
    ap.add_argument('--bed', help='BED file of regions (0-based, half-open)')
    ap.add_argument('--out', required=True, help='.tsv or .tsv.gz')
    ap.add_argument('--backbone', help='local NTv3 directory (default: the one recorded in the checkpoint)')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    regions = parse_regions(a)
    genome = read_fasta(a.fasta, {r[0] for r in regions} or None)
    regions = regions or [(c, 0, None) for c in genome]
    model = load_model(a.checkpoint, a.backbone, a.device)
    n = 0
    with (gzip.open(a.out, 'wt') if a.out.endswith('.gz') else open(a.out, 'w')) as f:
        f.write('chrom\tpos\tstrand\tcontext\tp\n')
        for chrom, start, end in regions:
            seq = genome[chrom]
            start, end = max(0, start), len(seq) if end is None else min(len(seq), end)
            for s in range(start, end, 2 ** 20):  # blocks are whole multiples of the 512-bp core
                e = min(end, s + 2 ** 20)
                p = predict_region(model, seq, s, e, batch=a.batch_size, device=a.device)
                lo = max(0, s - 2)
                ctx = contexts(seq[lo:e + 2])[:, s - lo:s - lo + e - s]
                h, i = np.nonzero(ctx)
                order = np.lexsort((h, i))
                h, i = h[order], i[order]
                pd.DataFrame(dict(chrom=chrom, pos=s + i + 1, strand=np.where(h == 0, '+', '-'),
                                  context=np.array(CONTEXTS)[ctx[h, i] - 1], p=p[h, i])
                             ).to_csv(f, sep='\t', header=False, index=False, float_format='%.5f')
                n += len(i)
    print(f'{n} cytosines written to {a.out}')


if __name__ == '__main__':
    main()
