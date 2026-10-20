"""Rewrite a REED checkpoint as weights plus model settings only, dropping any other saved metadata.

python convert_checkpoint.py original.pt reed_model.pt [--species arabidopsis_thaliana]
"""
import argparse

import torch

from reed.model import DEFAULT_BACKBONE, ROPE_KEYS

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('src')
ap.add_argument('dst')
ap.add_argument('--species', default='arabidopsis_thaliana', help='NTv3 species token used in training')
ap.add_argument('--backbone', default=DEFAULT_BACKBONE)
a = ap.parse_args()
blob = torch.load(a.src, map_location='cpu', weights_only=True)
sd = {k: v for k, v in blob.get('model', blob).items() if not k.endswith(ROPE_KEYS)}
torch.save(dict(model=sd, config=dict(backbone=a.backbone, species=a.species)), a.dst)
print(f'{len(sd)} tensors written to {a.dst}')
