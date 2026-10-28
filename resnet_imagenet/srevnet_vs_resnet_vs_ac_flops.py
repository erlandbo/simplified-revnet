import torch
from torch import nn

from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union


from opv1_w_backward import ResNetOpV1ModuleIterator
from opv2_w_backward import ResNetOpV2ModuleIterator
from op_recompute_activation import RecomputeActivationModuleIterator

def measure_flops(
    model: torch.nn.Module,
    forward_fn: Callable[[], torch.Tensor],
    loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> int:
    """Utility to compute the total number of FLOPs used by a module during training or during inference.

    It's recommended to create a meta-device model for this:

    Example::

        with torch.device("meta"):
            model = MyModel()
            x = torch.randn(2, 32)

        model_fwd = lambda: model(x)
        fwd_flops = measure_flops(model, model_fwd)

        model_loss = lambda y: y.sum()
        fwd_and_bwd_flops = measure_flops(model, model_fwd, model_loss)

    Args:
        model: The model whose FLOPs should be measured.
        forward_fn: A function that runs ``forward`` on the model and returns the result.
        loss_fn: A function that computes the loss given the ``forward_fn`` output. If provided, the loss and `backward`
            FLOPs will be included in the result.

    """
    from torch.utils.flop_counter import FlopCounterMode

    if loss_fn is None:
        flop_counter_fwd = FlopCounterMode(display=False, depth=None)
        with flop_counter_fwd:
            forward_fn()
        flop_counter_fwd.get_total_flops()
    else:
        flop_counter_fwd = FlopCounterMode(display=False, depth=None)
        flop_counter_bwd = FlopCounterMode(display=False, depth=None)
        with flop_counter_fwd:
            logits = forward_fn() 
        with flop_counter_bwd:
            loss = loss_fn(logits)
            loss.backward()

    return flop_counter_fwd.get_total_flops(), flop_counter_bwd.get_total_flops()



def conv_block(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU())

def feedforward_block(in_feats, out_feats):
    return nn.Sequential(
        nn.Linear(in_feats, out_feats),
        nn.ReLU(),
        nn.Linear(in_feats, out_feats)
    )


def run_opv1(input, functions, device):
    meta_model = ResNetOpV1ModuleIterator(nn.ModuleList(functions)).to(device)
    model_fwd = lambda: meta_model(input, torch.zeros_like(input))[0]  # noqa: F821
    model_loss = lambda y: y.mean()  # noqa: F821
    fwd_flops, bwd_flops = measure_flops(meta_model, model_fwd, model_loss)
    return fwd_flops, bwd_flops

def run_opv2(input, functions, device):
    meta_model = ResNetOpV2ModuleIterator(nn.ModuleList(functions)).to(device)
    model_fwd = lambda: meta_model(input, torch.zeros_like(input))[0]  # noqa: F821
    model_loss = lambda y: y.mean()  # noqa: F821
    fwd_flops, bwd_flops = measure_flops(meta_model, model_fwd, model_loss)
    return fwd_flops, bwd_flops

def run_resnet(input, functions, device):
    meta_model = nn.Sequential(*functions).to(device)
    model_fwd = lambda: meta_model(input)  # noqa: F821
    model_loss = lambda y: y.mean()  # noqa: F821
    fwd_flops, bwd_flops = measure_flops(meta_model, model_fwd, model_loss)
    return fwd_flops, bwd_flops

def run_resnet_ac(input, functions, device):
    meta_model = RecomputeActivationModuleIterator(nn.ModuleList(functions)).to(device)
    model_fwd = lambda: meta_model(input)  # noqa: F821
    model_loss = lambda y: y.mean()  # noqa: F821
    fwd_flops, bwd_flops = measure_flops(meta_model, model_fwd, model_loss)
    return fwd_flops, bwd_flops


device = "cuda:0"
batch_size = 4
channels = 32
depths = torch.arange(0, 400+1, 25)
feat_size = 224

arch = "resnet"
print(arch)


all_fwd_flops = []
all_bwd_flops = []

for depth in depths:
    if depth == 0: continue

    x = torch.randn((batch_size, channels, feat_size, feat_size), device=device, requires_grad=True)

    residual_modules =[conv_block(channels, channels) for _ in range(depth)]

    if arch == "opv1":
        fwd_flops, bwd_flops = run_opv1(x, residual_modules, device)
    elif arch == "opv2":
        fwd_flops, bwd_flops = run_opv2(x, residual_modules, device)
    elif arch == "resnet":
        fwd_flops, bwd_flops = run_resnet(x, residual_modules, device)
    elif arch == "resnet-ac":
        fwd_flops, bwd_flops = run_resnet_ac(x, residual_modules, device)

    all_fwd_flops.append(fwd_flops / 10**9)
    all_bwd_flops.append(bwd_flops / 10**9)

print(all_fwd_flops)
print(all_bwd_flops)
