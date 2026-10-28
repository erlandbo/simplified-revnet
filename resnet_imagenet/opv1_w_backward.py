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

class OpV1ModuleIteratorFunction(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        xi,
        xa,
        modules_F,
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

                tmp = modules_F[i](yi)
                
                ya.add_(tmp) 
                yi.add_(ya)
                del tmp
        
        del xi, xa
            
        ctx.save_for_backward(yi, ya)

        ctx.fwd_device_states = fwd_device_states

        return yi, ya


    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, *grad_outputs):
        dyi, dya = grad_outputs
        
        (yi, ya) = ctx.saved_tensors

        fwd_device_states = ctx.fwd_device_states

        modules_F = ctx.modules_F
        num_modules = len(modules_F)

        grads_weights_F = []

        yi = yi.detach().clone().requires_grad_(False)
        ya = ya.detach().clone().requires_grad_(False)
        #yi = yi.detach().requires_grad_(False)
        #ya = ya.detach().requires_grad_(False)

        for i in range(num_modules - 1, -1, -1):

            with torch.no_grad():
                yi.sub_(ya)  # (yi = xi + ya)

            with torch.enable_grad():
                yi.requires_grad_(True)

                # set seed to have correct dropout
                torch.manual_seed(fwd_device_states[i])

                F_xi = modules_F[i](yi)

                torch.autograd.backward(F_xi, dyi + dya)

            #yi.detach_().requires_grad_(False)
            F_xi = F_xi.detach()

            with torch.no_grad():            
                ya.sub_(F_xi)  # (ya = xa + F(xi))
                del F_xi

                #grads_weights_F.append(ddF[1:])
                dya.add_(dyi) # dxa = dyi + dya

                #dyi.add_(ddF[0])  # dxi = dyi + ddF[0]
                dyi.add_(yi.grad)  # dxi = dyi + ddF[0]
                yi.grad = None
                yi.detach_().requires_grad_(False)
                #del ddF

        with torch.no_grad():
            yi.detach_().requires_grad_(False)        
            ya.detach_().requires_grad_(False)        
            del yi, ya

        grads = (dyi, dya, None)
        
        return grads 


class ResNetOpV1ModuleIterator(nn.Module):
    
    def __init__(self, res_modules):
        super().__init__()
        self.res_modules = res_modules

        
    def forward(self, xi, xa):

        yi, ya = OpV1ModuleIteratorFunction.apply(
            xi,
            xa,
            self.res_modules,
        )

        return yi, ya


class ResNetOpV1ModuleIteratorWBackprop(nn.Module):
    
    def __init__(self, res_modules):
        super().__init__()
        self.res_modules = res_modules
        
    def forward(self, xi, xa):
        num_modules = len(self.res_modules)

        for i in range(0, num_modules):

            tmp = self.res_modules[i](xi)

            ya = xa + tmp
            yi = xi + ya

            xa = ya
            xi = yi

        return yi, ya
