import torch.nn as nn

__all__ = ["DarkNetBasebone", "DarkNet20", "build_darknet20"]

class DarkNetBasebone(nn.Module):
    def __init__(self, in_channels=3, active_fn=nn.ReLU(True)):
        super(DarkNetBasebone, self).__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            active_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 192, 3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            active_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.block_3 = nn.Sequential(
            nn.Conv2d(192, 128, 1, stride=1, padding=0),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.Conv2d(256, 256, 1, stride=1, padding=0),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            active_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.block_4 = nn.Sequential(
            nn.Conv2d(512, 256, 1, stride=1, padding=0),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.Conv2d(512, 256, 1, stride=1, padding=0),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.Conv2d(512, 256, 1, stride=1, padding=0),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.Conv2d(512, 512, 1, stride=1, padding=0),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            active_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.block_5 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, stride=1, padding=0),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.Conv2d(1024, 512, 1, stride=1, padding=0),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            active_fn,
        )
        
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

class DarkNet20(nn.Module):
    def __init__(self, in_channels=3, num_class=1000, active_fn=nn.ReLU(True)):
        super(DarkNet20, self).__init__()
        
        self.feature_block = DarkNetBasebone(in_channels, active_fn)
        
        self.classifer = nn.Sequential(
            nn.Linear(1024 * 4 * 4, 4096),
            nn.Dropout(),
            active_fn,
            nn.Linear(4096, num_class),
            
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
        output = self.classifer(output)
        
        return output
    
    def get_feature_block(self):
        return self.feature_block
        
def build_darknet20(pretrained=False, **kwargs):
   
    model = DarkNet20(**kwargs)
    if pretrained:
        pass
    return model