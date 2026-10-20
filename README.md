# REED

REED predicts the methylation level of every cytosine, on both strands and in all sequence contexts (CG, CHG, CHH), from DNA sequence alone. It fine-tunes the NTv3-100M post-trained DNA language model (InstaDeep) with a small per-base head, and reads each prediction from 10 kb of sequence around it. A model is trained on one genome and one methylome (or a consensus of many) and can then be applied to any sequence of that genome or another.

## Install

Python 3.10 and a GPU with 24 GB of memory (prediction also runs on CPU, slowly).

```
pip install -r requirements.txt
```

The backbone `InstaDeepAI/NTv3_100M_post` is downloaded from the Hugging Face Hub on first use, at the revision pinned in `configs/reed_100m.json`. To work offline, pass a local copy with `--backbone DIR`. The backbone's license (non-commercial use) also applies to models trained with REED.

## Train

```
python train.py --fasta genome.fa --methylation sample.CX_report.txt.gz --format cx \
    --train-chroms 1,2,3 --val-chroms 4 --out runs/my_model
```

Stage 1 trains the head with the backbone frozen. Stage 2 fine-tunes all parameters, starting from the best head. In both stages, the checkpoint with the lowest validation MSE (mean over CG, CHG and CHH) is kept, as `runs/my_model/head.pt` and `runs/my_model/reed.pt`. Chromosomes not named in `--train-chroms` or `--val-chroms` are never read, so they stay available for testing.

Training settings (`configs/reed_100m.json`):

- **Windows:** 10-kb windows are sampled uniformly from the training chromosomes, and half of them are reverse-complemented.
- **Loss:** mean squared error over the central 4,096 bp, at cytosines with coverage of at least 5. Each cytosine is weighted by min(coverage, 20)/20. The loss is normalised within each context, and the contexts are combined CG:CHG:CHH = 1:1:2.
- **Optimiser:** AdamW, batch 32, cosine schedule, gradient clipping at 1, full FP32 (TF32 off).
- **Head stage:** 3,000 steps, learning rate 1e-3.
- **Full stage:** 8,000 steps, learning rate 3e-5, warm-up 500 steps.
- **Validation:** 4,096 fixed windows on the validation chromosomes, scored on the central 512 bp of each.
- **Species token:** the backbone is conditioned on `arabidopsis_thaliana` for every plant (`species`).

To train on a consensus of many methylomes, give the mean methylation of each cytosine as a bedMethyl file, with the number of contributing samples as its coverage, and set `min_cov` to 20.

## Input formats

| `--format` | Columns |
|---|---|
| `cx` | Bismark CX report: chrom, position (1-based), strand, methylated count, unmethylated count, context, trinucleotide |
| `allc` | methylpy allc: chrom, position (1-based), strand, trinucleotide, methylated reads, total reads, ... |
| `bedmethyl` | chrom, start (0-based), end, name, score, strand, ..., coverage (column 10), percent methylated (column 11) |

Plain or gzipped files both work. Chromosome names must match the FASTA headers (first word). Contexts are taken from the reference. A row is used only if its position is a cytosine on the stated strand and, where the file gives a context, the context agrees with the reference. If fewer than 95% of rows agree, training stops, because that usually means a different assembly or coordinate base.

## Predict

```
python predict.py --checkpoint reed.pt --fasta genome.fa --region 4:1000001-1100000 --out pred.tsv.gz
```

`--region` takes `chrom` or `chrom:start-end` (1-based, inclusive) and can be repeated. `--bed` takes a BED file of regions. With neither option, the whole FASTA is predicted. The output is a tab-separated table with a header line and one row per cytosine: `chrom`, `pos` (1-based), `strand`, `context`, `p` (predicted methylation level, 0 to 1). Each 512-bp block is predicted from the 10 kb centred on it, and sequence beyond the chromosome ends is treated as N.

Checkpoints that also carry training metadata load as they are. `python convert_checkpoint.py in.pt out.pt` strips one down to the weights and model settings.

## Citation

Wang et al., in preparation.
