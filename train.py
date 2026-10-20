"""Train REED: stage 1 trains the head on the frozen backbone, stage 2 fine-tunes all parameters from the best head.

python train.py --fasta genome.fa --methylation sample.CX_report.txt.gz --format cx \
    --train-chroms 1,2,3 --val-chroms 4 --out runs/my_model
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from reed.data import CONTEXTS, FORMATS, get_batch, prepare_labels, sample_windows, val_anchors
from reed.model import REED, load_weights, weights


def lr_at(step, peak, warmup, steps):
    """Linear warm-up, then cosine decay to 5% of the peak."""
    progress = min(1., max(0., (step - warmup) / (steps - warmup)))
    return peak * min(1., step / max(1, warmup)) * (.05 + .95 * (1 + math.cos(math.pi * progress)) / 2)


def evaluate(model, labels, anchors, cfg, micro, device):
    """Unweighted MSE and Pearson r per context on the central eval_core bp of each validation window."""
    model.eval()
    W, core = cfg['window'], cfg['eval_core']
    mid = (W - core) // 2
    pred, obs = {i: [] for i in (1, 2, 3)}, {i: [] for i in (1, 2, 3)}
    with torch.no_grad():
        for i in range(0, len(anchors), micro):
            chroms, centers = zip(*anchors[i:i + micro])
            x, y, ctx, _ = get_batch(labels, chroms, centers, W, core)
            p = torch.sigmoid(model(x.to(device)))[:, :, mid:mid + core].cpu().numpy()
            for c in pred:
                pred[c].append(p[ctx.numpy() == c])
                obs[c].append(y.numpy()[ctx.numpy() == c])
    res = {}
    for c, name in enumerate(CONTEXTS, 1):
        p, y = np.concatenate(pred[c]).astype(float), np.concatenate(obs[c]).astype(float)
        if len(p):
            res[name] = dict(n=len(p), mse=float(np.mean((p - y) ** 2)), r=float(np.corrcoef(p, y)[0, 1]))
    res['macro_mse'] = float(np.mean([v['mse'] for v in res.values()]))
    return res


def run_stage(name, model, labels, train, anchors, cfg, st, path, device, eval_first):
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=st['lr'], weight_decay=cfg['weight_decay'])
    W, core, B, micro, cw = cfg['window'], cfg['train_core'], cfg['batch_size'], st['micro_batch'], cfg['context_weights']
    mid = (W - core) // 2
    best = math.inf

    def checkpoint(step):
        nonlocal best
        ev = evaluate(model, labels, anchors, cfg, micro, device)
        if ev['macro_mse'] < best:
            best = ev['macro_mse']
            torch.save(dict(model=weights(model, head_only=name == 'head'), step=step, config=cfg), path)
        print(json.dumps(dict(stage=name, step=step, validation=ev, best=best)), flush=True)

    if eval_first:
        checkpoint(0)
    for step in range(1, st['steps'] + 1):
        model.train()
        lr = lr_at(step, st['lr'], st['warmup'], st['steps'])
        for g in opt.param_groups:
            g['lr'] = lr
        chroms, centers, rc = sample_windows(rng, labels, train, B, W)
        x, y, ctx, w = get_batch(labels, chroms, centers, W, core, rc)
        # per-context normalisers over the whole batch, so micro-batch gradients add up to the batch gradient
        counts = [max(1e-6, float((w * (ctx == c)).sum())) for c in (1, 2, 3)]
        x, y, ctx, w = (t.to(device) for t in (x, y, ctx, w))
        opt.zero_grad(set_to_none=True)
        total = 0.
        for i in range(0, B, micro):
            p = torch.sigmoid(model(x[i:i + micro]))[:, :, mid:mid + core]
            err = (p - y[i:i + micro]).square() * w[i:i + micro]
            c = ctx[i:i + micro]
            loss = sum(a * (err * (c == k)).sum() / counts[k - 1] for k, a in enumerate(cw, 1)) / sum(cw)
            loss.backward()
            total += loss.item()
        if not math.isfinite(total):
            raise FloatingPointError(f'non-finite loss at {name} step {step}')
        torch.nn.utils.clip_grad_norm_(params, cfg['grad_clip'])
        opt.step()
        if step == 1 or step % cfg['log_every'] == 0:
            print(json.dumps(dict(stage=name, step=step, loss=total, lr=lr)), flush=True)
        if step % st['eval_every'] == 0 or step == st['steps']:
            checkpoint(step)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=str(Path(__file__).parent / 'configs' / 'reed_100m.json'))
    ap.add_argument('--fasta', required=True)
    ap.add_argument('--methylation', required=True, help='per-cytosine methylation table')
    ap.add_argument('--format', required=True, choices=FORMATS)
    ap.add_argument('--train-chroms', required=True, help='comma-separated, e.g. 1,2,3')
    ap.add_argument('--val-chroms', required=True, help='comma-separated, used for checkpoint selection')
    ap.add_argument('--out', required=True)
    ap.add_argument('--backbone', help='local NTv3 directory (default: the hub model in the config)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if a.backbone:
        cfg.update(backbone=a.backbone, backbone_revision=None)
    torch.backends.cuda.matmul.allow_tf32 = False  # full FP32 throughout
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(cfg['seed'])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train, val = a.train_chroms.split(','), a.val_chroms.split(',')
    labels = prepare_labels(a.fasta, a.methylation, a.format, train + val, out / 'labels', cfg['min_cov'], cfg['cov_cap'])
    W = cfg['window']
    train = [c for c in train if len(labels[c]['seq']) > 2 * W]
    anchors = val_anchors(labels, val, cfg['val_windows'], cfg['eval_core'], W, cfg['seed'])
    model = REED(cfg['backbone'], cfg.get('backbone_revision'), cfg['species'], cfg['head_hidden']).to(a.device)

    model.freeze_backbone(True)
    run_stage('head', model, labels, train, anchors, cfg, cfg['head_stage'], out / 'head.pt', a.device, False)
    load_weights(model, torch.load(out / 'head.pt', map_location='cpu', weights_only=True)['model'], head_only=True)
    model.freeze_backbone(False)
    run_stage('full', model, labels, train, anchors, cfg, cfg['full_stage'], out / 'reed.pt', a.device, True)
    print('saved', out / 'reed.pt')


if __name__ == '__main__':
    main()
