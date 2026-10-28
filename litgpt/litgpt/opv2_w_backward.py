import torch
from torch import nn

import sys

# OpV2LoopFunction
# dxi = dL/dyi dyi/dxi + dL/dya dya/dxi 
# dxa = dL/dyi dyi/dxa + dL/dya dya/dxa

# New version, using single for-loop.
# https://github.com/renmengye/revnet-public/blob/master/resnet/models/revnet_model.py

# ya = xa + xi
# yi = xi + modules_G[i](ya)

# dxi = dL/dyi dyi/dxi + dL/dya dya/dxi 
# dxa = dL/dyi dyi/dxa + dL/dya dya/dxa

# dxi = dL/dyi (dxi/dxi + dya/dxi) + dL/dya (dxa/dxi + dxi/dxi) 
#     = dL/dyi (dxi/dxi + dG_ya/dxi) + dL/dya (dxa/dxi + dxi/dxi)
#     = dL/dyi (1 + dG_ya/dxi) + dL/dya (0 + 1)
#     = dL/dyi (1 + dG_ya/dya dya/dxi) + dL/dya (0 + 1)
#     = dL/dyi (1 + dG_ya/dya dxi/dxi) + dL/dya (0 + 1)
#     = dL/dyi (1 + dG_ya/dya) + dL/dya (0 + 1)
#     = dL/dyi + dL/dyi dG_ya/dya) + dL/dya 

# dxa = dL/dyi (dxi/dxa + dya/dxa) + dL/dya (dxa/dxa + dxi/dxa) 
#     = dL/dyi (dxi/dxa + dG_ya/dxa) + dL/dya (dxa/dxa + dxi/dxa)
#     = dL/dyi (0 + dG_ya/dya dya/dxa) + dL/dya (1 + 0)
#     = dL/dyi (dG_ya/dya (dxa/dxa + dxi/dxa)) + dL/dya
#     = dL/dyi (dG_ya/dya (1 + 0)) + dL/dya
#     = dL/dyi dG_ya/dya + dL/dya

# dG  = dL/dyi dyi/dG + dL/dya dya/G
#     = dL/dyi (dxi/dG + dG/dG) + dL/dya (dxa/G + dxi/dG)
#     = dL/dyi (0 + dG/dG) + dL/dya (0 + dxi/dG)
#     = dL/dyi dG/dG



class OpV2ModuleIteratorFunction(torch.autograd.Function):

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
        modules_G,
    ):
        ctx.modules_G = modules_G

        num_modules = len(modules_G)

        fwd_device_states = []

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

                ya.add_(yi)   # ya = xa + xi
                # tmp = modules_G[i](ya)

                if rope_indices is not None:
                    tmp = modules_G[i](ya, cos[..., rope_indices[i // 2]], sin[..., rope_indices[i // 2]], mask, input_pos, input_pos_maxp1 )
                else:
                    tmp = modules_G[i](ya, cos, sin, mask, input_pos, input_pos_maxp1 )

                yi.add_(tmp)  #yi = xi + modules_G[i](ya)
                del tmp

        ctx.save_for_backward(yi, ya)

        ctx.fwd_device_states = fwd_device_states

        ctx.cos = cos
        ctx.sin = sin
        ctx.mask = mask
        ctx.input_pos = input_pos
        ctx.input_pos_maxp1 = input_pos_maxp1
        ctx.rope_indices = rope_indices

        return yi, ya


    @staticmethod
    def backward(ctx, *grad_outputs):
        dyi, dya = grad_outputs
        
        yi, ya = ctx.saved_tensors

        modules_G = ctx.modules_G
        num_modules = len(modules_G)

        fwd_device_states = ctx.fwd_device_states

        cos = ctx.cos
        sin = ctx.sin
        mask = ctx.mask
        input_pos = ctx.input_pos
        input_pos_maxp1 = ctx.input_pos_maxp1
        rope_indices = ctx.rope_indices


        yi = yi.detach().clone().requires_grad_(False)
        ya = ya.detach().clone().requires_grad_(False)

        for i in range(num_modules - 1, -1, -1):
        
            with torch.set_grad_enabled(True):
                ya.requires_grad_(True)

                 # set seed to have correct dropout
                torch.manual_seed(fwd_device_states[i])
               
                #G_ya = modules_G[i](ya)

                if rope_indices is not None:
                    G_ya = modules_G[i](ya, cos[..., rope_indices[i // 2]], sin[..., rope_indices[i // 2]], mask, input_pos, input_pos_maxp1 )
                else:
                    G_ya = modules_G[i](ya, cos, sin, mask, input_pos, input_pos_maxp1)

                torch.autograd.backward(G_ya, dyi)
                # ddG = torch.autograd.grad(G_ya, (ya, ) + tuple(modules_G[i].parameters()), dyi)

            ya.detach_().requires_grad_(False)
            G_ya = G_ya.detach()

            with torch.no_grad():
                yi.sub_(G_ya)  # xi = yi - G_ya
                del G_ya
                ya.sub_(yi)    #xa = ya - xi
 
                # dxi = dyi + ddG[0] + dya
                #dyi.add_(ddG[0])
                dyi.add_(ya.grad)

                dyi.add_(dya)
 
                # dxa = ddG[0] + dya
                #dya.add_(ddG[0])
                dya.add_(ya.grad)
                ya.grad = None

                #del ddG

        with torch.no_grad():
            yi.detach_().requires_grad_(False)        
            ya.detach_().requires_grad_(False)        
            del yi, ya

        grads = (dyi, dya, None, None, None, None, None, None, None)
        
        return grads 


class ResNetOpV2ModuleIterator(nn.Module):
    
    def __init__(self, res_modules):
        super().__init__()
        self.res_modules = res_modules
        
    def forward(self, xi, xa, cos, sin, mask, input_pos, input_pos_maxp1, rope_indices):
        # module_weights = [p for res_module in self.res_modules for p in res_module.parameters() ]

        yi, ya = OpV2ModuleIteratorFunction.apply(
            xi,
            xa,
            cos, 
            sin, 
            mask, 
            input_pos, 
            input_pos_maxp1,
            rope_indices,
            self.res_modules,
        )

        return yi, ya


class ResNetOpV2ModuleIteratorWBackprop(nn.Module):
    
    def __init__(self, res_modules):
        super().__init__()
        self.res_modules = res_modules
        
    def forward(self, xi, xa, cos, sin, mask, input_pos, input_pos_maxp1, rope_indices):
        num_modules = len(self.res_modules)

        for i in range(0, num_modules):

            ya = xa + xi

            if rope_indices is not None:
                tmp = self.res_modules[i](ya, cos[..., rope_indices[i // 2]], sin[..., rope_indices[i // 2]], mask, input_pos, input_pos_maxp1 )
            else:
                tmp = self.res_modules[i](ya, cos, sin, mask, input_pos, input_pos_maxp1 )

            yi = xi + tmp

            xa = ya
            xi = yi

        return yi, ya

