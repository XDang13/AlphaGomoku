import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_conv2d import BasicConv2d

__all__ = ["InceptionBlockA", "InceptionBlockB", "InceptionBlockC",
           "InceptionBlockD", "InceptionBlockE", "InceptionV3Basebone",
           "InceptionV3", "build_inceptionv3"]

class InceptionBlockA(nn.Module):

    def __init__(self, in_channels, pool_features, active_fn=nn.ReLU(True)):
        super(InceptionBlockA, self).__init__()
        self.block_1x1 = BasicConv2d(in_channels, 64, kernel_size=1, active_fn=active_fn)

        self.block_5x5 = nn.Sequential(
            BasicConv2d(in_channels, 48, kernel_size=1, active_fn=active_fn),
            BasicConv2d(48, 64, kernel_size=5, padding=2, active_fn=active_fn),
        )
        
        self.block_3x3dbl = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1, active_fn=active_fn),
            BasicConv2d(64, 96, kernel_size=3, padding=1, active_fn=active_fn),
            BasicConv2d(96, 96, kernel_size=3, padding=1, active_fn=active_fn),
        )

        self.block_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_features, kernel_size=1, active_fn=active_fn),
        )
        

    def forward(self, x):
        block_1x1_output = self.block_1x1(x)

        block_5x5_output = self.block_5x5(x)

        block_3x3dbl_output = self.block_3x3dbl(x)
        
        block_pool_output = self.block_pool(x)

        outputs = [block_1x1_output, block_5x5_output, block_3x3dbl_output, block_pool_output]
        return torch.cat(outputs, 1)

class InceptionBlockB(nn.Module):

    def __init__(self, in_channels, active_fn=nn.ReLU(True)):
        super(InceptionBlockB, self).__init__()
        self.block_3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2, active_fn=nn.ReLU(True))

        self.block_3x3dbl = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1, active_fn=active_fn),
            BasicConv2d(64, 96, kernel_size=3, padding=1, active_fn=active_fn),
            BasicConv2d(96, 96, kernel_size=3, stride=2, active_fn=active_fn),
        )
        
        self.block_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        block_3x3_output = self.block_3x3(x)

        block_3x3dbl_output = self.block_3x3dbl(x)

        block_pool_output = self.block_pool(x)

        outputs = [block_3x3_output, block_3x3dbl_output, block_pool_output]
        return torch.cat(outputs, 1)

class InceptionBlockC(nn.Module):

    def __init__(self, in_channels, channels_7x7, active_fn=nn.ReLU(True)):
        super(InceptionBlockC, self).__init__()
        self.block_1x1 = BasicConv2d(in_channels, 192, kernel_size=1, active_fn=active_fn)
        
        
        self.block_7x7 = nn.Sequential(
            BasicConv2d(in_channels, channels_7x7, kernel_size=1, active_fn=active_fn),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3), active_fn=active_fn),
            BasicConv2d(channels_7x7, 192, kernel_size=(7, 1), padding=(3, 0), active_fn=active_fn),
        )
        
        self.block_7x7bdl = nn.Sequential(
            BasicConv2d(in_channels, channels_7x7, kernel_size=1, active_fn=active_fn),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0), active_fn=active_fn),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3), active_fn=active_fn),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0), active_fn=active_fn),
            BasicConv2d(channels_7x7, 192, kernel_size=(1, 7), padding=(0, 3), active_fn=active_fn),
        )

        

        self.block_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 192, kernel_size=1, active_fn=active_fn),
        )

    def forward(self, x):
        block_1x1_output = self.block_1x1(x)

        block_7x7_output = self.block_7x7(x)

        block_7x7bdl_output = self.block_7x7bdl(x)

        block_pool_output = self.block_pool(x)

        outputs = [block_1x1_output, block_7x7_output, block_7x7bdl_output, block_pool_output]
        return torch.cat(outputs, 1)

class InceptionBlockD(nn.Module):

    def __init__(self, in_channels, active_fn=nn.ReLU(True)):
        super(InceptionBlockD, self).__init__()
        self.block_3x3 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1, active_fn=active_fn),
            BasicConv2d(192, 320, kernel_size=3, stride=2, active_fn=active_fn),
        )

        self.block_7x7x3 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1, active_fn=active_fn),
            BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3), active_fn=active_fn),
            BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0), active_fn=active_fn),
            BasicConv2d(192, 192, kernel_size=3, stride=2, active_fn=active_fn),
        )
        
        self.block_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        block_3x3_output = self.block_3x3(x)

        block_7x7x3_output = self.block_7x7x3(x)
        

        block_pool_output = self.block_pool(x)
        outputs = [block_3x3_output, block_7x7x3_output, block_pool_output]
        return torch.cat(outputs, 1)

class InceptionBlockE(nn.Module):

    def __init__(self, in_channels, active_fn=nn.ReLU(True)):
        super(InceptionBlockE, self).__init__()
        self.block_1x1 = BasicConv2d(in_channels, 320,
                                     kernel_size=1, active_fn=active_fn)

        self.block_3x3 = BasicConv2d(in_channels, 384, kernel_size=1, active_fn=active_fn)
            
        self.block_3x3_list = nn.ModuleList(
            [
                BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1), active_fn=active_fn), 
                BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0), active_fn=active_fn),
            ]
        )

        self.block_3x3dbl = nn.Sequential(
            BasicConv2d(in_channels, 448, kernel_size=1, active_fn=active_fn),
            BasicConv2d(448, 384, kernel_size=3, padding=1, active_fn=active_fn),
        )
        
        self.block_3x3dbl_list = nn.ModuleList(
            [
                BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1), active_fn=active_fn),
                BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0), active_fn=active_fn),
            ]
        )

        self.block_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 192, kernel_size=1, active_fn=active_fn),
        )

    def forward(self, x):
        block_1x1_ouput = self.block_1x1(x)

        block_3x3_ouput = self.block_3x3(x)
        block_3x3_ouput = torch.cat([layer(block_3x3_ouput) for layer in self.block_3x3_list],
                                    1)

        block_3x3dbl_output = self.block_3x3dbl(x)
                
        block_3x3dbl_output = torch.cat([layer(block_3x3dbl_output) for layer in self.block_3x3dbl_list],
                                 1)
        block_pool_output = self.block_pool(x)

        outputs = [block_1x1_ouput, block_3x3_ouput, block_3x3dbl_output, block_pool_output]
        return torch.cat(outputs, 1)

class InceptionBlockAux(nn.Module):

    def __init__(self, in_channels, num_classes, active_fn=nn.ReLU(True)):
        super(InceptionBlockAux, self).__init__()
        self.block_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.block_1 = BasicConv2d(in_channels, 128, kernel_size=1, active_fn=active_fn)
        self.block_2 = BasicConv2d(128, 768, kernel_size=5, active_fn=active_fn)
        self.block_2.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        output = self.block_pool(x)
        output = self.block_1(output)
        output = self.block_2(output)
        
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        
        return output

class InceptionV3Basebone(nn.Module):
    
    def __init__(self, in_channels=3, aux_logits=True, active_fn=nn.ReLU(True)):
        super(InceptionV3Basebone, self).__init__()
        
        self.aux_logits = aux_logits
        
        self.block_1 = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2, active_fn=active_fn),
            BasicConv2d(32, 32, kernel_size=3, active_fn=active_fn),
            BasicConv2d(32, 64, kernel_size=3, padding=1, active_fn=active_fn),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.block_2 = nn.Sequential(
            BasicConv2d(64, 80, kernel_size=1, active_fn=active_fn),
            BasicConv2d(80, 192, kernel_size=3, active_fn=active_fn),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.block_3 = nn.Sequential(
            InceptionBlockA(192, pool_features=32, active_fn=active_fn),
            InceptionBlockA(256, pool_features=64, active_fn=active_fn),
            InceptionBlockA(288, pool_features=64, active_fn=active_fn),
        )
        
        self.block_4 = nn.Sequential(
            InceptionBlockB(288, active_fn=active_fn),
            InceptionBlockC(768, channels_7x7=128, active_fn=active_fn),
            InceptionBlockC(768, channels_7x7=160, active_fn=active_fn),
            InceptionBlockC(768, channels_7x7=160, active_fn=active_fn),
            InceptionBlockC(768, channels_7x7=192, active_fn=active_fn),
        )
            
        self.block_5 = nn.Sequential(
            InceptionBlockD(768, active_fn=active_fn),
            InceptionBlockE(1280, active_fn=active_fn),
            InceptionBlockE(2048, active_fn=active_fn),
        )
        
    def forward(self, x):
        output = self.block_1(x)
        output = self.block_2(output)
        output = self.block_3(output)
        output = self.block_4(output)
        if self.training and self.aux_logits:
            aux = output

        output = self.block_5(output)

        if self.training and self.aux_logits:
            return output, aux

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

class InceptionV3(nn.Module):

    def __init__(self, in_channels=3, num_classes=1000, aux_logits=True,
                 active_fn=nn.ReLU(True)):
        super(InceptionV3, self).__init__()
        self.aux_logits = aux_logits
        
        self.feature_block = InceptionV3Basebone(in_channels, aux_logits, active_fn)
        
        if self.aux_logits:
            self.block_aux = InceptionBlockAux(768, num_classes, active_fn=active_fn)
        
        self.block_pool = nn.AvgPool2d(kernel_size=8)
        
        self.fc = nn.Linear(2048, num_classes)
        
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


    def forward(self, x):
        if self.training and self.aux_logits:
            output, aux = self.feature_block(x)
            aux = self.block_aux(aux)
        else:
            output = self.feature_block(x)
        
        output = self.block_pool(output)
        output = F.dropout(output, training=self.training)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        
        if self.training and self.aux_logits:
            return output, aux
        
        return output
    
    def get_feature_block(self):
        return self.feature_block

def build_inceptionv3(pretrained=False, **kwargs):
    
    model = InceptionV3(**kwargs)
    if pretrained:
        pass
        
    return model
