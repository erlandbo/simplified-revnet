import torch
from torch import nn
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union
import resnet_torchvision as resnet_models
import reversible_resnet_torchvision as reversible_models
import revnet_torchvision_v3 as revnet_models
import argparse



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

    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        if loss_fn is None:
            forward_fn()
        else:
            loss_fn(forward_fn()).backward()
    return flop_counter.get_total_flops()


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--method', default="opv1", type=str, help='method', choices=["opv1", "opv2", "mresnet", "revnet", "resnet-ac", "resnet"])
args = parser.parse_args()

if args.method == "opv1":
    arch = "reversible_resnet101"
    recompute_op = "reversible-opv1"
elif args.method == "opv2":
    arch = "reversible_resnet101"
    recompute_op = "reversible-opv2"
elif args.method == "mresnet":
    arch = "reversible_resnet101"
    recompute_op = "reversible-momentum"
elif args.method == "revnet":
    arch = "revnet104"
    recompute_op = "reversible-revnet"
elif args.method == "resnet-ac":
    arch = "resnet101"
    recompute_op = "recompute-activations"
elif args.method == "resnet":
    arch = "resnet101"
    recompute_op = None

print("arch: ", arch)
print("recompute_op: ", recompute_op)

device = "cuda:0"
batch_size = 256
world_size = 2

with torch.device("meta"):
    if recompute_op is None:
        meta_model = resnet_models.__dict__[arch](recompute_activations=False, use_backprop=False)
    elif recompute_op == "recompute-activations":
        meta_model = resnet_models.__dict__[arch](recompute_activations=True, use_backprop=False)
    elif recompute_op == "reversible-revnet":
        meta_model = revnet_models.__dict__[arch](use_backprop=False) 
    else:
        meta_model = reversible_models.__dict__[arch](recompute_op=recompute_op, gamma=0.9, use_backprop=False)        
    
    cross_entropy = nn.CrossEntropyLoss().to(device)

    x = torch.randn((batch_size, 3, 224, 224))
    y = torch.randint(0, 1000, (batch_size,))
    model_fwd = lambda: meta_model(x)  # noqa: F821
    model_loss = lambda logits: cross_entropy(logits, y)  # noqa: F821
    measured_flops = measure_flops(meta_model, model_fwd, model_loss)
    print(f"Measured TFLOPs: {measured_flops * world_size / 1e12:.2f}")
