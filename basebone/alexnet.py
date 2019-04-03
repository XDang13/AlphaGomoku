import torch.nn as nn
from .basic_conv2d import BasicConv2d

__all__ = ["AlexNetBasebone", "AlexNet", "build_alexnet"]

class AlexNetBasebone(nn.Module):
    def __init__(self, in_channels=3, active_fn=nn.ReLU(True)):
        super(AlexNetBasebone, self).__init__()
        
        self.block_1 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=11,
                         stride=4, padding=2, bn=False, active_fn=active_fn),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.block_2 = nn.Sequential(
            BasicConv2d(64, 192, kernel_size=5, padding=2,
                         bn=False, active_fn=active_fn),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.block_3 = nn.Sequential(
            BasicConv2d(192, 384, kernel_size=3, padding=1,
                         bn=False, active_fn=active_fn),
            BasicConv2d(384, 256, kernel_size=3, padding=1,
                         bn=False, active_fn=active_fn),
            BasicConv2d(256, 256, kernel_size=3, padding=1,
                         bn=False, active_fn=active_fn),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
    def forward(self, x):
        output = self.block_1(x)
        output = self.block_2(output)
        output = self.block_3(output)
        
        return output
    
    @property
    def to_list(self):
        return [self.block_1, self.block_2, self.block_3]
    
    @property
    def to_sequential(self):
        
        return nn.Sequential(*self.to_list)
    
    @property
    def to_modellist(self):
        
        return nn.ModuleList(self.to_list)
    
class AlexNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000,
                 active_fn=nn.ReLU(True)):
        super(AlexNet, self).__init__()
        
        self.feature_block = AlexNetBasebone(in_channels, active_fn)
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            active_fn,
            nn.Dropout(),
            nn.Linear(4096, 4096),
            active_fn,
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
        
def build_alexnet(pretrain=False, **kwargs):
    model = AlexNet(**kwargs)
    if pretrain:
        pass
    
    return model
        