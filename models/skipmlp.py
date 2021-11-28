import torch
import torch.nn as nn

from models import register


@register('skipmlp')
class SkipMLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, skips=[]):
        super().__init__()
        self.hidden_list = hidden_list
        self.skips = skips
        lastv = in_dim
        for i, hidden in enumerate(hidden_list):
            if i in skips:
                layer = nn.Sequential(
                    nn.Linear(lastv + in_dim, hidden),
                    nn.ReLU()
                )
            else:
                layer = nn.Sequential(
                    nn.Linear(lastv, hidden),
                    nn.ReLU()
                )
            setattr(self, f"encoding_{i}", layer)
            lastv = hidden
        self.encoding_final = nn.Linear(lastv, out_dim)
        self.out_dim = out_dim

    def forward(self, inp):
        shape = inp.shape[:-1]
        inp = inp.view(-1, inp.shape[-1])
        x = inp
        for i in range(len(self.hidden_list)):
            if i in self.skips:
                x = torch.cat([inp, x], dim=-1)
            x = getattr(self, f"encoding_{i}")(x)
        x = self.encoding_final(x)
        return x.view(*shape, -1)

@register('skipmlp-dim')
class SkipMLPDim(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, skip_dim=None, skips=[]):
        super().__init__()
        self.hidden_list = hidden_list
        self.skips = skips
        lastv = in_dim
        skip_dim = skip_dim or in_dim

        for i, hidden in enumerate(hidden_list):
            if i in skips:
                layer = nn.Sequential(
                    nn.Linear(lastv + skip_dim, hidden),
                    nn.ReLU()
                )
            else:
                layer = nn.Sequential(
                    nn.Linear(lastv, hidden),
                    nn.ReLU()
                )
            setattr(self, f"encoding_{i}", layer)
            lastv = hidden

        if len(self.hidden_list) in self.skips:
            self.encoding_final = nn.Linear(lastv + skip_dim, out_dim)
        else:
            self.encoding_final = nn.Linear(lastv, out_dim)

        self.out_dim = out_dim
        self.skip_dim = skip_dim

    def forward(self, inp):
        shape = inp.shape[:-1]
        inp = inp.view(-1, inp.shape[-1])
        sk = inp[:, -self.skip_dim:]
        x = inp

        for i in range(len(self.hidden_list)):
            if i in self.skips:
                x = torch.cat([x, sk], dim=-1)
            x = getattr(self, f"encoding_{i}")(x)

        if len(self.hidden_list) in self.skips:
            x = torch.cat([x, sk], dim=-1)
        x = self.encoding_final(x)
        return x.view(*shape, -1)
