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
    #@custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(
        ctx,
        x_init,
        cos,
        sin,
        mask,
        input_pos,
        input_pos_maxp1,
        rope_indices,
        modules_F,
    ):
        fwd_device_states = []
        ctx.modules_F = modules_F

        num_modules = len(modules_F)

        activations = []

        with torch.no_grad():
            x = x_init.clone().detach().contiguous()

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

                if rope_indices is not None:
                    x = modules_F[i](x, cos[..., rope_indices[i // 2]], sin[..., rope_indices[i // 2]], mask, input_pos, input_pos_maxp1 )
                else:
                    x = modules_F[i](x, cos, sin, mask, input_pos, input_pos_maxp1 )
        
        ctx.save_for_backward(*activations)

        ctx.fwd_device_states = fwd_device_states

        ctx.cos = cos
        ctx.sin = sin
        ctx.mask = mask
        ctx.input_pos = input_pos
        ctx.input_pos_maxp1 = input_pos_maxp1
        ctx.rope_indices = rope_indices

        return x


    @staticmethod
    #@custom_bwd(device_type='cuda')
    def backward(ctx, *grad_outputs):
        (dy,) = grad_outputs
        
        activations = ctx.saved_tensors

        fwd_device_states = ctx.fwd_device_states

        cos = ctx.cos
        sin = ctx.sin
        mask = ctx.mask
        input_pos = ctx.input_pos
        input_pos_maxp1 = ctx.input_pos_maxp1
        rope_indices = ctx.rope_indices

        modules_F = ctx.modules_F
        num_modules = len(modules_F)

        for i in range(num_modules - 1, -1, -1):

            x = activations[i].detach().requires_grad_(False)

            with torch.enable_grad():
                x.requires_grad_(True)

                # set seed to have correct dropout
                torch.manual_seed(fwd_device_states[i])

                if rope_indices is not None:
                    F_xi = modules_F[i](x, cos[..., rope_indices[i // 2]], sin[..., rope_indices[i // 2]], mask, input_pos, input_pos_maxp1 )
                else:
                    F_xi = modules_F[i](x, cos, sin, mask, input_pos, input_pos_maxp1)

                torch.autograd.backward(F_xi, dy)

            #yi.detach_().requires_grad_(False)
            F_xi = F_xi.detach()

            with torch.no_grad():            
                del F_xi
                dy = x.grad

                x.grad = None
                x.detach_().requires_grad_(False)
                del x

        grads = (dy, None, None, None, None, None, None, None)
        
        return grads 


class RecomputeActivationModuleIterator(nn.Module):
    
    def __init__(self, res_modules):
        super().__init__()
        self.res_modules = res_modules
        
    def forward(self, xi, xa, cos, sin, mask, input_pos, input_pos_maxp1, rope_indices):
        res_modules = list(self.res_modules)

        y  = RecomputeActivationModuleIteratorFunction.apply(
            xi,
            cos, 
            sin, 
            mask, 
            input_pos, 
            input_pos_maxp1,
            rope_indices,
            res_modules,
        )

        return y, xa


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
        
    def forward(self, xi, xa, cos, sin, mask, input_pos, input_pos_maxp1, rope_indices):

        num_modules = len(self.res_modules)

        for i in range(0, num_modules):

            if rope_indices is not None:
                xi = self.res_modules[i](xi, cos[..., rope_indices[i // 2]], sin[..., rope_indices[i // 2]], mask, input_pos, input_pos_maxp1 )
            else:
                xi = self.res_modules[i](xi, cos, sin, mask, input_pos, input_pos_maxp1 )

        return xi, xa
