"""REED: the NTv3 post-trained DNA language model with a per-base methylation head."""
import torch
from torch import nn
from transformers import AutoModel

DEFAULT_BACKBONE = 'InstaDeepAI/NTv3_100M_post'
NTV3_TOKENS = [6, 8, 9, 7, 10]  # A, C, G, T, N -> NTv3 vocabulary ids
ROPE_KEYS = ('.rotary_embedding.cos_cached', '.rotary_embedding.sin_cached')  # rebuilt on the fly


class REED(nn.Module):
    """codes (B, L) int64, A0 C1 G2 T3 N4 -> logits (B, 2, L).
    Channel 0 is the plus-strand cytosine at each position, channel 1 the minus-strand cytosine (a reference G)."""

    def __init__(self, backbone=DEFAULT_BACKBONE, revision=None, species='arabidopsis_thaliana', head_hidden=256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone, revision=revision, trust_remote_code=True).float()
        sid = torch.as_tensor(self.backbone.encode_species([species])).reshape(-1)[:1].long()
        self.register_buffer('species_id', sid, persistent=False)
        self.register_buffer('lut', torch.tensor(NTV3_TOKENS), persistent=False)
        dim = self.backbone.config.embed_dim
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, head_hidden), nn.GELU(), nn.Linear(head_hidden, 2))
        self.frozen = False

    def freeze_backbone(self, frozen=True):
        self.frozen = frozen
        self.backbone.requires_grad_(not frozen)
        return self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def features(self, tok):
        B, L = tok.shape
        pad = (-L) % 128  # seven 2x down-sampling stages; pad symmetrically with N, crop afterwards
        left = pad // 2
        if pad:
            n = int(self.lut[4])
            tok = torch.cat([tok.new_full((B, left), n), tok, tok.new_full((B, pad - left), n)], 1)
        core = self.backbone.core
        # parent forward of the post-trained core: the full-length embedding, without the track heads
        out = super(type(core), core).forward(input_ids=tok, condition_ids=[self.species_id.expand(B)],
                                              conditions_masks=None, output_hidden_states=False,
                                              output_attentions=False)
        return out['embedding'][:, left:left + L].float()

    def forward(self, codes):
        tok = self.lut[codes]
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            feat = self.features(tok)
        return self.head(feat).transpose(1, 2)


def weights(model, head_only=False):
    sd = {k: v for k, v in model.state_dict().items() if not k.endswith(ROPE_KEYS)}
    return {k: v for k, v in sd.items() if k.startswith('head.')} if head_only else sd


def load_weights(model, sd, head_only=False):
    sd = {k: v for k, v in sd.items() if not k.endswith(ROPE_KEYS)}
    if head_only:
        return model.head.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith('head.')})
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing + unexpected if not k.endswith(ROPE_KEYS)]
    if bad:
        raise RuntimeError(f'checkpoint does not match the model: {bad[:5]}')


def load_model(path, backbone=None, device='cpu'):
    """Trained REED model in eval mode. `backbone` overrides the checkpoint's (e.g. a local copy)."""
    blob = torch.load(path, map_location='cpu', weights_only=True)
    sd, cfg = blob.get('model', blob), blob.get('config', {})
    model = REED(backbone or cfg.get('backbone', DEFAULT_BACKBONE), None if backbone else cfg.get('backbone_revision'),
                 cfg.get('species', 'arabidopsis_thaliana'), sd['head.1.weight'].shape[0])
    load_weights(model, sd)
    return model.to(device).eval()
