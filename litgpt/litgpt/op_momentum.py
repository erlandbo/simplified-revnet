import torch
from torch import nn

import sys

from torch.amp import custom_fwd, custom_bwd

from litgpt.exact_rep_pytorch import TorchExactRep


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

# See gradient derivations https://github.com/michaelsdr/momentumnet/blob/main/momentumnet/momentum_net.py

class OpMomentumModuleIteratorFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        xi,
        xa,
        cos,
        sin,
        mask,
        input_pos,
        input_pos_maxp1,
        rope_indices,
        modules_F,
        gamma,
    ):
        fwd_device_states = []
        ctx.modules_F = modules_F

        num_modules = len(modules_F)

        with torch.no_grad():
            yi = xi.clone().detach().contiguous()
            ya = xa.clone().detach().contiguous()

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

                if rope_indices is not None:
                    tmp = modules_F[i](yi, cos[..., rope_indices[i // 2]], sin[..., rope_indices[i // 2]], mask, input_pos, input_pos_maxp1 )
                else:
                    tmp = modules_F[i](yi, cos, sin, mask, input_pos, input_pos_maxp1 )
                
                ya *= gamma
                ya += tmp * (1 - gamma)  # ya = m * xa + (1-m) * F 
                yi.add_(ya)
                del tmp
        
        del xi, xa
            
        ctx.save_for_backward(yi, ya)

        ctx.fwd_device_states = fwd_device_states

        ctx.cos = cos
        ctx.sin = sin
        ctx.mask = mask
        ctx.input_pos = input_pos
        ctx.input_pos_maxp1 = input_pos_maxp1
        ctx.rope_indices = rope_indices

        ctx.gamma = gamma

        return yi, ya


    @staticmethod
    def backward(ctx, *grad_outputs):
        dyi, dya = grad_outputs
        
        (yi, ya) = ctx.saved_tensors

        fwd_device_states = ctx.fwd_device_states

        cos = ctx.cos
        sin = ctx.sin
        mask = ctx.mask
        input_pos = ctx.input_pos
        input_pos_maxp1 = ctx.input_pos_maxp1
        rope_indices = ctx.rope_indices

        gamma = ctx.gamma

        modules_F = ctx.modules_F
        num_modules = len(modules_F)

        grads_weights_F = []

        yi = yi.detach().clone().requires_grad_(False)
        ya = ya.detach().clone().requires_grad_(False)

        for i in range(num_modules - 1, -1, -1):

            with torch.no_grad():
                yi.sub_(ya)  # (yi = xi + ya)

            with torch.enable_grad():
                yi.requires_grad_(True)

                # set seed to have correct dropout
                torch.manual_seed(fwd_device_states[i])

                if rope_indices is not None:
                    F_xi = modules_F[i](yi, cos[..., rope_indices[i // 2]], sin[..., rope_indices[i // 2]], mask, input_pos, input_pos_maxp1 )
                else:
                    F_xi = modules_F[i](yi, cos, sin, mask, input_pos, input_pos_maxp1)

                torch.autograd.backward(F_xi, (dyi + dya) * (1-gamma))

            F_xi = F_xi.detach()

            with torch.no_grad():
                ya += -(1 - gamma) * F_xi  # ya = m*xa + (1-m)*F  <=> xa = (ya - (1-m)F) / m
                ya /= gamma 

                del F_xi

                dya.add_(dyi) # dxa = m * (dyi + dya)
                dya.mul_(gamma)

                dyi.add_(yi.grad)  # dxi = dyi + ddF[0] * (1-gamma)
                yi.grad = None
                yi.detach_().requires_grad_(False)
        
        with torch.no_grad():
            yi.detach_().requires_grad_(False)        
            ya.detach_().requires_grad_(False)        
            del yi, ya

        grads = (dyi, dya, None, None, None, None, None, None, None, None)
        
        return grads 


class ResNetOpMomentumModuleIterator(nn.Module):
    
    def __init__(self, res_modules, gamma):
        super().__init__()
        self.res_modules = res_modules
        self.gamma = gamma

        
    def forward(self, xi, xa, cos, sin, mask, input_pos, input_pos_maxp1, rope_indices):
        res_modules = list(self.res_modules)
        # module_weights = [p for res_module in self.res_modules for p in res_module.parameters() ]
        # module_weights = [p for res_module in res_modules for p in res_module.parameters() ]
        # import pdb; pdb.set_trace()

        yi, ya = OpMomentumModuleIteratorFunction.apply(
            xi,
            xa,
            cos, 
            sin, 
            mask, 
            input_pos, 
            input_pos_maxp1,
            rope_indices,
            res_modules,
            self.gamma
        )

        return yi, ya


class ResNetOpV1(nn.Module):
    
    def __init__(self, simplified_reversible_modules):
        super().__init__()
        self.simplified_reversible_modules = simplified_reversible_modules
        
    def forward(self, xi, xa):
        for network_module in self.simplified_reversible_modules: 
            xi, xa = network_module(xi, xa)
        return xi, xa


class ResNetOpMomentumModuleIteratorWBackprop(nn.Module):
    
    def __init__(self, res_modules, gamma):
        super().__init__()
        self.res_modules = res_modules
        self.gamma = gamma
        
    def forward(self, xi, xa, cos, sin, mask, input_pos, input_pos_maxp1, rope_indices):

        num_modules = len(self.res_modules)

        for i in range(0, num_modules):

            if rope_indices is not None:
                tmp = self.res_modules[i](xi, cos[..., rope_indices[i // 2]], sin[..., rope_indices[i // 2]], mask, input_pos, input_pos_maxp1 )
            else:
                tmp = self.res_modules[i](xi, cos, sin, mask, input_pos, input_pos_maxp1 )

            ya = self.gamma * xa + (1.0 - self.gamma) * tmp
            yi = xi + ya

            xa = ya
            xi = yi

        return yi, ya
