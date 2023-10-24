from timm.layers.norm_act import BatchNormAct2d, _create_act
from torch.nn import functional as F
import torch 
import torch.nn as nn

class GroupNormAct2d(nn.GroupNorm):
    """GroupNorm + Activation"""
    def __init__(
            self,
            num_groups, 
            num_channels,
            eps=1e-5,
            act_layer=nn.SiLU,
            act_kwargs=None,
            apply_act=True,
            inplace=True,
            weight= None, 
            bias= None,
            drop_layer=None
    ):

        super(GroupNormAct2d, self).__init__(
            num_groups = num_groups, 
            num_channels = num_channels,
            eps=eps
        )
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act)

    def forward(self, x):
        # cut & paste of torch.nn.BatchNorm2d.forward impl to avoid issues with torchscript and tracing
        x = F.group_norm(
            x,
            self.num_groups, 
            self.weight, 
            self.bias,
            self.eps
        )
        x = self.drop(x)
        x = self.act(x)
        return x

# num_channels: num_groups
GROUP_NORM_LOOKUP = {
        16: 2,     # -> channels per group: 8
        32: 4,     # -> channels per group: 8
        64: 8,     # -> channels per group: 8
        96: 16,    # -> channels per group: 6
        128: 8,    # -> channels per group: 16
        192: 12,   # -> channels per group: 16
        256: 16,   # -> channels per group: 16
        320: 20,   # -> channels per group: 16 
        512: 32,   # -> channels per group: 16
        384:16,    # -> channels per group: 24
        768: 24,   # -> channels per group: 32
        1024: 32,  # -> channels per group: 32
        2048: 32,  # -> channels per group: 64
    }

def coatnet_to_group_norm(module: torch.nn.Module):

    for name, child in module.named_children():

        if isinstance(child, BatchNormAct2d):
            num_channels = child.num_features
            setattr(module, name, GroupNormAct2d(GROUP_NORM_LOOKUP[num_channels], num_channels))
            
        elif isinstance(child, nn.BatchNorm2d):
            child: nn.BatchNorm2d = child
            num_channels = child.num_features
            setattr(module, name, torch.nn.GroupNorm(GROUP_NORM_LOOKUP[num_channels], num_channels))

        else:
            coatnet_to_group_norm(child)


def poolformer_to_group_norm(module: torch.nn.Module):

    for name, child in module.named_children():
            
        if isinstance(child, nn.GroupNorm):
            child: torch.nn.GroupNorm = child
            num_channels = child.num_channels
            setattr(module, name, torch.nn.GroupNorm(GROUP_NORM_LOOKUP[num_channels], num_channels))

        else:
            poolformer_to_group_norm(child)