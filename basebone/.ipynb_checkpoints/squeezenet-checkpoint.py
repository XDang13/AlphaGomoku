import torch
import torch.nn as nn

__all__ = ["SqueezeBlock", "SqueezeNetBaseboneV1_0", "SqueezeNetBaseboneV1_1",
           "SqueezeNet", "build_squeezenet1_0", "build_squeezenet1_1"]

class SqueezeBlock(nn.Module):

    def __init__(self, in_channels, channels,
                 expand1x1_channels, expand3x3_channels,
                 active_fn=nn.ReLU(True)):
        super(SqueezeBlock, self).__init__()
        
        self.block_squeeze = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1),
            active_fn,
        )
        
        self.block_expand1x1 = nn.Sequential(
            nn.Conv2d(channels, expand1x1_channels,
                                   kernel_size=1),
            active_fn,
        )
        
        self.block_expand3x3 = nn.Sequential(
            nn.Conv2d(channels, expand3x3_channels,
                                   kernel_size=3, padding=1),
            active_fn,
        )

    def forward(self, x):
        output = self.block_squeeze(x)
        output_expand1x1 = self.block_expand1x1(output)
        output_expand3x3 = self.block_expand3x3(output)
        output = torch.cat([output_expand1x1, output_expand3x3], 1)
        
        return output
    
class SqueezeNetBaseboneV1_0(nn.Module):

    def __init__(self, in_channels=3, active_fn=nn.ReLU(True)):
        super(SqueezeNetBaseboneV1_0, self).__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=7, stride=2),
            active_fn,
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )

        self.block_2 = nn.Sequential(
            SqueezeBlock(96, 16, 64, 64, active_fn=active_fn),
            SqueezeBlock(128, 16, 64, 64, active_fn=active_fn),
            SqueezeBlock(128, 32, 128, 128, active_fn=active_fn),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )

        self.block_3 = nn.Sequential(
            SqueezeBlock(256, 32, 128, 128, active_fn=active_fn),
            SqueezeBlock(256, 48, 192, 192, active_fn=active_fn),
            SqueezeBlock(384, 48, 192, 192, active_fn=active_fn),
            SqueezeBlock(384, 64, 256, 256, active_fn=active_fn),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )

        self.block_4 = nn.Sequential(
            SqueezeBlock(512, 64, 256, 256, active_fn=active_fn),
        )
        
    def forward(self, x):
        output = self.block_1(x)
        output = self.block_2(output)
        output = self.block_3(output)
        output = self.block_4(output)
        
        return output
    
    @property
    def to_list(self):
        return [self.block_1, self.block_2, self.block_3,
                self.block_4]
    
    @property
    def to_sequential(self):
        
        return nn.Sequential(*self.to_list)
    
    @property
    def to_modellist(self):
        
        return nn.ModuleList(self.to_list)

class SqueezeNetBaseboneV1_1(nn.Module):

    def __init__(self, in_channels=3, active_fn=nn.ReLU(True)):
        super(SqueezeNetBaseboneV1_1, self).__init__()
        
        self.block_1 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=2),
                active_fn,
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )

        self.block_2 = nn.Sequential(
                SqueezeBlock(64, 16, 64, 64, active_fn=active_fn),
                SqueezeBlock(128, 16, 64, 64, active_fn=active_fn),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )

        self.block_3 = nn.Sequential(
                SqueezeBlock(128, 32, 128, 128, active_fn=active_fn),
                SqueezeBlock(256, 32, 128, 128, active_fn=active_fn),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )

        self.block_4 = nn.Sequential(
                SqueezeBlock(256, 48, 192, 192, active_fn=active_fn),
                SqueezeBlock(384, 48, 192, 192, active_fn=active_fn),
                SqueezeBlock(384, 64, 256, 256, active_fn=active_fn),
                SqueezeBlock(512, 64, 256, 256, active_fn=active_fn),
        )
        
    def forward(self, x):
        output = self.block_1(x)
        output = self.block_2(output)
        output = self.block_3(output)
        output = self.block_4(output)
        
        return output
    
    @property
    def to_list(self):
        return [self.block_1, self.block_2, self.block_3,
                self.block_4]
    
    @property
    def to_sequential(self):
        
        return nn.Sequential(*self.to_list)
    
    @property
    def to_modellist(self):
        
        return nn.ModuleList(self.to_list)
    
class SqueezeNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=1000,
                 version=1.0, active_fn=nn.ReLU(True)):
        super(SqueezeNet, self).__init__()
        
        assert version in [1.0, 1.1]
        
        if version == 1.0:
            self.feature_block = SqueezeNetBaseboneV1_0(in_channels, active_fn)
        else:
            self.feature_block = SqueezeNetBaseboneV1_1(in_channels, active_fn)
            
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
    def forward(self, x):
        output = self.feature_block(x)
        
        output = self.classifier(output)
        
        output = output.view(output.size(0), -1)
        
        return output
    
    def get_feature_block(self):
        return self.feature_block
    
def build_squeezenet1_0(pretrained=False, **kwargs):
    model = SqueezeNet(version=1.0, **kwargs)
    if pretrained:
        pass
    return model

def build_squeezenet1_1(pretrained=False, **kwargs):
    model = SqueezeNet(version=1.1, **kwargs)
    if pretrained:
        pass
    return model