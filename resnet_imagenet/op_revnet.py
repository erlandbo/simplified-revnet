import torch
from torch import nn

import sys

from torch.amp import custom_fwd, custom_bwd

# New version, using single for-loop.
# https://github.com/renmengye/revnet-public/blob/master/resnet/models/revnet_model.py

# See gradient derivation https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reformer_pytorch.py

class OpRevNetModuleIteratorFunction(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x,
        modules_F,
        modules_G,
    ):
        fwd_device_states_F = []
        fwd_device_states_G = []
        ctx.modules_F = modules_F
        ctx.modules_G = modules_G

        num_modules = len(modules_F)

        xa, xi = torch.chunk(x, 2, dim=1)

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
                    fwd_state_F = torch.cuda.default_generators[device_idx].seed()
                else:
                    # CPU
                    fwd_state_F = int(torch.seed() % sys.maxsize)

                torch.manual_seed(fwd_state_F)
                fwd_device_states_F.append(fwd_state_F)

                # xa ~ x1, xi ~ x2 
                F_xi = modules_F[i](yi)
                
                ya.add_(F_xi)
                del F_xi 

                if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
                    # GPU
                    device_idx = torch.cuda.current_device()
                    fwd_state_G = torch.cuda.default_generators[device_idx].seed()
                else:
                    # CPU
                    fwd_state_G = int(torch.seed() % sys.maxsize)

                torch.manual_seed(fwd_state_G)
                fwd_device_states_G.append(fwd_state_G)

                # MLP, only ya required
                G_xa = modules_G[i](ya)

                yi.add_(G_xa)
                del G_xa
        
        del xi, xa, x

        y = torch.cat([ya, yi], dim=1)
        
        ctx.save_for_backward(y)

        ctx.fwd_device_states_F = fwd_device_states_F
        ctx.fwd_device_states_G = fwd_device_states_G

        return y


    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, *grad_outputs):
        (dy,) = grad_outputs
        dya, dyi = torch.chunk(dy, 2, dim=1)
        del dy

        (y, ) = ctx.saved_tensors

        fwd_device_states_F = ctx.fwd_device_states_F
        fwd_device_states_G = ctx.fwd_device_states_G

        modules_F = ctx.modules_F
        modules_G = ctx.modules_G
        num_modules = len(modules_F)

        # grads_weights_F = []
        # grads_weights_G = []

        ya, yi = torch.chunk(y, 2, dim=1)
        del y

        yi = yi.detach().clone().requires_grad_(False)
        ya = ya.detach().clone().requires_grad_(False)

        for i in range(num_modules - 1, -1, -1):

            with torch.set_grad_enabled(True):
                ya.requires_grad_(True)

                # set seed to have correct dropout
                torch.manual_seed(fwd_device_states_G[i])
               
                G_ya = modules_G[i](ya)

                torch.autograd.backward(G_ya, dyi)  # dL/dyi dG(ya)/dya

            ya.detach_().requires_grad_(False)
            G_ya = G_ya.detach()

            with torch.no_grad():
                yi.sub_(G_ya)  # xi = yi - G_ya
                del G_ya

                dya.add_(ya.grad)
                ya.grad = None

            ############################

            with torch.enable_grad():
                yi.requires_grad_(True)

                # set seed to have correct dropout
                torch.manual_seed(fwd_device_states_F[i])

                F_xi = modules_F[i](yi)

                torch.autograd.backward(F_xi, dya, retain_graph=True)  # dL/dyi + dL/

            F_xi = F_xi.detach()

            with torch.no_grad():            
                ya.sub_(F_xi)  # (ya = xa + F(xi))
                del F_xi

                dyi.add_(yi.grad)  # dxi = dyi + ddF[0]
                yi.grad = None
                yi.detach_().requires_grad_(False)

        with torch.no_grad():
            yi.detach_().requires_grad_(False)        
            ya.detach_().requires_grad_(False)        
            
            y = torch.cat([ya, yi], dim=1)
            dy = torch.cat([dya, dyi], dim=1)
            
            del yi, ya, dya, dyi, y

        grads = (dy, None, None)
        
        return grads 


class ResNetOpRevNetModuleIterator(nn.Module):
    
    def __init__(self, res_modules):
        super().__init__()
        F_modules, G_modules = zip(*res_modules)
        self.F_modules = nn.ModuleList(F_modules)
        self.G_modules = nn.ModuleList(G_modules)
        
    def forward(self, xi):

        y = OpRevNetModuleIteratorFunction.apply(
            xi,
            self.F_modules,
            self.G_modules,
        )

        return y



class ResNetOpRevNetModuleIteratorWBackprop(nn.Module):
    
    def __init__(self, res_modules):
        super().__init__()
        F_modules, G_modules = zip(*res_modules)
        self.F_modules = nn.ModuleList(F_modules)
        self.G_modules = nn.ModuleList(G_modules)

    def forward(self, xi):
        num_modules = len(self.F_modules)

        xa, xi = torch.chunk(xi, 2, dim=1)

        for i in range(0, num_modules):

            F_xi = self.F_modules[i](xi)

            ya = xa + F_xi
            
            G_xa = self.G_modules[i](ya)
            
            yi = xi + G_xa

            xa = ya
            xi = yi

        y = torch.cat([ya, yi], dim=1)

        return y



class ResNetOpDownsampleRevNetModuleWBackprop(nn.Module):
    
    def __init__(self, F_module, G_module, downsample_xa, downsample_xi):
        super().__init__()
        self.F_module, self.G_module, self.downsample_xa, self.downsample_xi = F_module, G_module, downsample_xa, downsample_xi

    def forward(self, xi):

        xa, xi = torch.chunk(xi, 2, dim=1)

        xa_ = self.downsample_xa(xa)
        xi_ = self.downsample_xi(xi)
        
        F_xi = self.F_module(xi)

        ya = xa_ + F_xi
        
        G_ya = self.G_module(ya)
        
        yi = xi_ + G_ya

        y = torch.cat([ya, yi], dim=1)

        return y
