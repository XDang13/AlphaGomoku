import torch.nn as nn
from .basic_conv2d import BasicConv2d

__all__ = ["VGGBasebone", "DefaultVGGBasebone", "VGG", "build_vgg16", "build_vgg19"]

class VGGBasebone(nn.Module):
    
    def _make_layer(self, cfg, bn=True, pool=True, bias=False):
        layers = []
        for channels in cfg:
            layers.append(BasicConv2d(self.in_channels, channels,
                                      kernel_size=3, padding=1,
                                      bn=bn, active_fn=self.active_fn,
                                      bias=bias))
            self.in_channels = channels
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))    
        
        return nn.Sequential(*layers)
    
class DefaultVGGBasebone(VGGBasebone):
    def __init__(self, cfg, in_channels=3, bn=True, active_fn=nn.ReLU(True)):
        super(DefaultVGGBasebone, self).__init__()
        self.in_channels = in_channels
        self.active_fn = active_fn
        self.block_1 = self._make_layer(cfg[0], bn=bn)
        self.block_2 = self._make_layer(cfg[1], bn=bn)
        self.block_3 = self._make_layer(cfg[2], bn=bn)
        self.block_4 = self._make_layer(cfg[3], bn=bn)
        self.block_5 = self._make_layer(cfg[4], bn=bn)
        
    def forward(self, x):
        
        output = self.block_1(x)
        output = self.block_2(output)
        output = self.block_3(output)
        output = self.block_4(output)
        output = self.block_5(output)
        
        return output
    
    @property
    def to_list(self):
        return [self.block_1, self.block_2, self.block_3,
                self.block_4, self.block_5]
    
    @property
    def to_sequential(self):
        
        return nn.Sequential(*self.to_list)
    
    @property
    def to_modellist(self):
        
        return nn.ModuleList(self.to_list)
    
class VGG(nn.Module):
    def __init__(self, cfg, in_channels=3, num_classes=1000,
                 bn=True, active_fn=nn.ReLU(True)):
        super(VGG, self).__init__()
        
        self.feature_block = DefaultVGGBasebone(cfg, in_channels,
                                                bn, active_fn)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            active_fn,
            nn.Dropout(),
            nn.Linear(4096, 4096),
            active_fn,
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        output = self.feature_block(x)
            
        output = output.view(output.size(0), -1)    
        output = self.classifier(output)
        
        return output
    
    def get_feature_block(self):
        return self.feature_block
            
def build_vgg16(pretrained=False, bn=True, **kwargs):
    cfg = [[64,64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]]
    model = VGG(cfg, bn=bn, **kwargs)
    if pretrained:
        pass
    return model
    
def build_vgg19(pretrained=False, bn=True, **kwargs):
    cfg = [[64,64], [128, 128], [256, 256, 256, 256],
           [512, 512, 512, 512], [512, 512, 512, 512]]
    model = VGG(cfg, bn=bn, **kwargs)
    if pretrained:
        pass
    return model