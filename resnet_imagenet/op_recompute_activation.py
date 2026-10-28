import torch
from torch import nn

import sys

from torch.amp import custom_fwd, custom_bwd


# OpV1LoopFunction
# dxi = dL/dyi dyi/dxi + dL/dya dya/dxi 
# dxa = dL/dyi dyi/dxa + dL/dya dya/dxa

# dxi = dL/dyi (dxi/dxi + dya/dxi) + dL/dya (dxa/dxi + dF/dxi) 
#     = dL/dyi (dxi/dxi + dF/dxi) + dL/dya (dxa/dxi + dF/dxi)
#     = dL/dyi (1 + dF/dxi) + dL/dya (0 + dF/dxi)
#     = dL/dyi * 1 + dL/dyi * dF/dxi + dL/dya * dF/dxi

# dxa = dL/dyi (dxi/dxa + dya/dxa) + dL/dya (dxa/dxa + dF/dxa) 
#     = dL/dyi (dxi/dxa + (dxa/dxa + dF/dxa)) + dL/dya (dxa/dxa + dF/dxa)
#     = dL/dyi (0 + (1 + 0)) + dL/dya (1 + 0)
#     = dL/dyi  + dL/dya

# dF  = dL/dyi dyi/dF + dL/dya dya/F
#     = dL/dyi (dxi/dF + dya/dF) + dL/dya (dxa/F + dF/dF)
#     = dL/dyi (dxi/dF + (dxa/dF + dF/dF))) + dL/dya (dxa/F + dF/dF)
#     = dL/dyi (0 + (0 + dF/dF))) + dL/dya (0 + dF/dF)
#     = dL/dyi  dF/dF + dL/dya dF/dF


# New version, using single for-loop.
# https://github.com/renmengye/revnet-public/blob/master/resnet/models/revnet_model.py

class RecomputeActivationModuleIteratorFunction(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x,
        modules_F,
    ):
        fwd_device_states = []
        ctx.modules_F = modules_F

        num_modules = len(modules_F)

        activations = []

        with torch.no_grad():
            x = x.clone().detach().contiguous()

            for i in range(0, num_modules):
                
                # https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/reformer/modeling_reformer.py#L1833
                # randomize seeds
                # use cuda generator if available
                if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
                    # GPU
                    device_idx = torch.cuda.current_device()
                    fwd_state = torch.cuda.default_generators[device_idx].seed()
                else:
                    # CPU
                    fwd_state = int(torch.seed() % sys.maxsize)

                torch.manual_seed(fwd_state)
                fwd_device_states.append(fwd_state)

                activations.append(x)

                x = modules_F[i](x)
        
        ctx.save_for_backward(*activations)

        ctx.fwd_device_states = fwd_device_states

        return x


    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, *grad_outputs):
        (dy,) = grad_outputs
        
        activations = ctx.saved_tensors

        fwd_device_states = ctx.fwd_device_states

        modules_F = ctx.modules_F
        num_modules = len(modules_F)

        for i in range(num_modules - 1, -1, -1):

            x = activations[i].detach().requires_grad_(False)

            with torch.enable_grad():
                x.requires_grad_(True)

                # set seed to have correct dropout
                torch.manual_seed(fwd_device_states[i])

                F_xi = modules_F[i](x)

                torch.autograd.backward(F_xi, dy)

            # yi.detach_().requires_grad_(False)
            F_xi = F_xi.detach()

            with torch.no_grad():            
                del F_xi
                dy = x.grad

                x.grad = None
                x.detach_().requires_grad_(False)
                del x

        grads = (dy, None)
        
        return grads 


class RecomputeActivationModuleIterator(nn.Module):
    
    def __init__(self, res_modules):
        super().__init__()
        self.res_modules = res_modules
        
    def forward(self, xi):
        res_modules = list(self.res_modules)

        y  = RecomputeActivationModuleIteratorFunction.apply(
            xi,
            res_modules,
        )

        return y


class ResNetOpV1(nn.Module):
    
    def __init__(self, simplified_reversible_modules):
        super().__init__()
        self.simplified_reversible_modules = simplified_reversible_modules
        
    def forward(self, xi, xa):
        for network_module in self.simplified_reversible_modules: 
            xi, xa = network_module(xi, xa)
        return xi, xa


class ResNetRecomputeActivationModuleIteratorWBackprop(nn.Module):
    
    def __init__(self, res_modules):
        super().__init__()
        self.res_modules = res_modules
        
    def forward(self, xi):

        num_modules = len(self.res_modules)

        for i in range(0, num_modules):

            xi = self.res_modules[i](xi)

        return xi
