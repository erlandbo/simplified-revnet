import torch
from torch import nn

from opv1_w_backward import ResNetOpV1ModuleIterator
from opv2_w_backward import ResNetOpV2ModuleIterator
from op_recompute_activation import RecomputeActivationModuleIterator

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
    net = ResNetOpV1ModuleIterator(nn.ModuleList(functions)).to(device)
    y, ya = net(input, torch.zeros_like(input))
    loss = y.mean()
    loss.backward()

def run_opv2(input, functions, device):
    net = ResNetOpV2ModuleIterator(nn.ModuleList(functions)).to(device)
    y, ya = net(input, torch.zeros_like(input))
    loss = y.mean()
    loss.backward()

def run_resnet(input, functions, device):
    net = nn.Sequential(*functions).to(device)
    y = net(input)
    loss = y.mean()
    loss.backward()

def run_resnet_ac(input, functions, device):
    net = RecomputeActivationModuleIterator(nn.ModuleList(functions)).to(device)
    y = net(input)
    loss = y.mean()
    loss.backward()


device = "cuda:0"
batch_size = 4
channels = 32
depths = torch.arange(0, 400+1, 25)
feat_size = 224

arch = "resnet-ac"
print(arch)

mem_usages = []

for depth in depths:
    if depth == 0: continue
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn((batch_size, channels, feat_size, feat_size), device=device, requires_grad=True)

    residual_modules =[conv_block(channels, channels) for _ in range(depth)]

    if arch == "opv1":
        run_opv1(x, residual_modules, device)
    elif arch == "opv2":
        run_opv2(x, residual_modules, device)
    elif arch == "resnet":
        run_resnet(x, residual_modules, device)
    elif arch == "resnet-ac":
        run_resnet_ac(x, residual_modules, device)

    max_mem_allocated = torch.cuda.max_memory_allocated(device) / 1024**3

    mem_usages.append(max_mem_allocated)

print(mem_usages)
