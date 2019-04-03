import torch.nn as nn
from .basic_conv2d import BasicConv2d

__all__ = ["BasicBlock", "Bottleneck", "ResNet", "ResNetBasebone", "build_resnet18",
           "build_resnet34", "build_resnet50", "build_resnet101",
           "build_resnet152"]

class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_channels, channels, stride=1,
                 downsample=None, active_fn=nn.ReLU(True)):
        super(BasicBlock, self).__init__()

        self.block_1 = BasicConv2d(in_channels, channels, kernel_size=1,
                                   active_fn=active_fn, bias=False)
        
        self.block_2 = BasicConv2d(channels, channels, kernel_size=3,
                                   stride=stride, padding=1, active_fn=None,
                                   bias=False)
        
        self.active_fn = active_fn
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        output = self.block_1(x)
 
        output = self.block_2(output)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        output += residual
        output = self.active_fn(output)
 
        return output

class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, in_channels, channels, stride=1,
                 downsample=None, active_fn=nn.ReLU(True)):
        super(Bottleneck, self).__init__()
        self.block_1 = BasicConv2d(in_channels, channels, kernel_size=1,
                                   active_fn=active_fn, bias=False)
        
        self.block_2 = BasicConv2d(channels, channels, kernel_size=3,
                                   stride=stride, padding=1, active_fn=active_fn,
                                   bias=False)
        
        self.block_3 = BasicConv2d(channels, channels*self.expansion, kernel_size=1,
                                   active_fn=None, bias=False)
        self.active_fn = active_fn
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        output = self.block_1(x)
 
        output = self.block_2(output)
         
        output = self.block_3(output)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        output += residual
        output = self.active_fn(output)
 
        return output

class ResNetBasebone(nn.Module):
    
    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                BasicConv2d(self.in_channels, channels * block.expansion,
                            kernel_size=1, stride=stride, bias=False, active_fn=None)
            )
 
        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample, self.active_fn))
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, active_fn=self.active_fn))
 
        return nn.Sequential(*layers)
        
class DefaultResNetBasebone(ResNetBasebone):
    def __init__(self, block, layers, in_channels=3, active_fn=nn.ReLU(True)):
        super(DefaultResNetBasebone, self).__init__()
        self.in_channels = 64
        
        self.active_fn = active_fn
        
        self.block_1 =nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=7, bn=True,
                        active_fn=active_fn, stride=2, padding=3, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.block_2 = self._make_layer(block, 64, layers[0])
        self.block_3 = self._make_layer(block, 128, layers[1], stride=2)
        self.block_4 = self._make_layer(block, 256, layers[2], stride=2)
        self.block_5 = self._make_layer(block, 512, layers[3], stride=2)
        
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
    
class ResNet(nn.Module):
 
    def __init__(self, block, layers, in_channels=3,
                 num_classes=1000, active_fn=nn.ReLU(True)):
        super(ResNet, self).__init__()
        
        self.feature_block = DefaultResNetBasebone(block, layers,
                                                   in_channels=in_channels,
                                                   active_fn=active_fn)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
 
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
 
    def forward(self, x):
        output = self.feature_block(x)
        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
 
        return output

    def get_feature_block(self):
        return self.feature_block
 
        
def build_resnet18(pretrained=False, **kwargs):
    
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pass
    return model


def build_resnet34(pretrained=False, **kwargs):
   
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pass
    return model


def build_resnet50(pretrained=False, **kwargs):
    
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pass
    return model


def build_resnet101(pretrained=False, **kwargs):
    
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pass
    return model


def build_resnet152(pretrained=False, **kwargs):
    
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pass
    return model