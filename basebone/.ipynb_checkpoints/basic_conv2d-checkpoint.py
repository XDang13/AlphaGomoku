import torch.nn as nn

__all__ = ["BasicConv2d"]

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 bn=True, active_fn=nn.ReLU(True), **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.active_fn = active_fn
        
    def forward(self, x):
        output = self.conv(x)
        if self.bn:
            output = self.bn(output)
        if self.active_fn:
            output = self.active_fn(output)
        return output