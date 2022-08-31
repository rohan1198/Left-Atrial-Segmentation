import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary


def passthrough(x):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)

        # self.bn1 = torch.nn.BatchNorm3d(nchan)

    def forward(self, x):
        #out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu1(self.conv1(x))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, in_channels, elu, attention_module = False):
        super(InputTransition, self).__init__()
        self.attention_module = attention_module
        self.num_features = 8
        self.in_channels = in_channels

        self.conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=5, padding=2)

        #self.bn1 = torch.nn.BatchNorm3d(self.num_features)

        self.relu1 = ELUCons(elu, self.num_features)

        if self.attention_module:
            self.att_module = AttentionModule(self.num_features)

    def forward(self, x):
        out = self.relu1(self.conv1(x))

        if self.attention_module:
            out = self.att_module(out)

        #repeat_rate = int(self.num_features / self.in_channels)
        #out = self.bn1(out)
        #x16 = x.repeat(1, repeat_rate, 1, 1, 1)
        #return self.relu1(torch.add(out, x16))

        out = torch.add(out, x)
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False, attention_module = False):
        super(DownTransition, self).__init__()
        self.attention_module = attention_module
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        #self.bn1 = torch.nn.BatchNorm3d(outChans)

        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        #self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

        if self.attention_module:
            self.att_module = AttentionModule(outChans)

    def forward(self, x):
        #down = self.relu1(self.bn1(self.down_conv(x)))
        down = self.relu1(self.down_conv(x))
        out = self.do1(down)
        out = self.ops(out)
        #out = self.relu2(torch.add(out, down))

        if self.attention_module:
            out = self.att_module(out)

        out = torch.add(out, down)
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)

        #self.bn1 = torch.nn.BatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        #skipxdo = self.do2(skipx)
        #out = self.relu1(self.bn1(self.up_conv(out)))
        #xcat = torch.cat((out, skipxdo), 1)
        #out = self.ops(xcat)
        #out = self.relu2(torch.add(out, xcat))
        #return out
        skip = skipx
        out = self.relu1(self.up_conv(out))
        out_cat = torch.cat((out, skip), 1)
        out = self.ops(out_cat)
        out = torch.add(out, out_cat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, in_channels, classes, elu):
        super(OutputTransition, self).__init__()
        self.classes = classes
        self.conv1 = nn.Conv3d(in_channels, classes, kernel_size=5, padding=2)
        #self.bn1 = torch.nn.BatchNorm3d(classes)

        #self.conv2 = nn.Conv3d(classes, classes, kernel_size=1)
        self.relu1 = ELUCons(elu, classes)

    def forward(self, x):
        # convolve 32 down to channels as the desired classes
        #out = self.relu1(self.bn1(self.conv1(x)))
        #out = self.conv2(out)
        out = self.relu1(self.conv1(x))
        return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 4):
        super(ChannelGate, self).__init__()
        #   
        self.channel_gate_layers = []
        self.channel_gate_layers.append(Flatten())
        self.channel_gate_layers.append(nn.Linear(in_channels, in_channels // reduction_ratio))
        self.channel_gate_layers.append(nn.BatchNorm1d(in_channels // reduction_ratio))
        self.channel_gate_layers.append(nn.ReLU())
        self.channel_gate_layers.append(nn.Linear(in_channels // reduction_ratio,in_channels))
        self.channel_gate = nn.Sequential(*self.channel_gate_layers)[0]

    def forward(self, x):
        avg_pool_out = F.avg_pool3d(x, kernel_size = (x.size(2),x.size(3),x.size(4)))
        x_out = self.channel_gate(avg_pool_out)
        c_att = torch.sigmoid(x_out).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        out_att = x * c_att
        return x + out_att

class SpatialGate(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 4, dilated_conv_layers = 2, dilation = 4):
        super(SpatialGate, self).__init__()
        self.spatial_gate_layers = []
        self.spatial_gate_layers.append(nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size = 1))
        self.spatial_gate_layers.append(nn.BatchNorm3d(in_channels // reduction_ratio))
        self.spatial_gate_layers.append(nn.ReLU())
        for i in range(dilated_conv_layers):
            self.spatial_gate_layers.append(nn.Conv3d(in_channels // reduction_ratio, in_channels // reduction_ratio, kernel_size = 3, 
                                            padding = dilation, dilation = dilation))
            self.spatial_gate_layers.append(nn.BatchNorm3d(in_channels // reduction_ratio))
            self.spatial_gate_layers.append(nn.ReLU())
        self.spatial_gate_layers.append(nn.Conv3d(in_channels // reduction_ratio, 1, kernel_size = 1))
        self.spatial_gate = nn.Sequential(*self.spatial_gate_layers)
    
    def forward(self, x):
        x_out =  self.spatial_gate(x)
        s_att = torch.sigmoid(x_out).expand_as(x)
        out_att = x * s_att
        return x + out_att

class AttentionModule(nn.Module):
    """
    Attention mechansim specified in "Park, J. et al. (2019) ‘BAM: Bottleneck attention module’, British Machine Vision Conference 2018, BMVC 2018."
    """
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.channel_att = ChannelGate(in_channels)
        self.spatial_att = SpatialGate(in_channels)
    
    def forward(self,x):
        x_out = self.channel_att(x)
        x_out = self.spatial_att(x_out)
        return x + x_out

"""
class VNetAttention(nn.Module):
    def __init__(self, elu=False, in_channels=1, classes=1, attention_module = True):
        super(VNetAttention, self).__init__()
        self.classes = classes
        self.in_channels = in_channels

        self.in_tr = InputTransition(in_channels, elu=elu, attention_module = attention_module)
        self.down_tr32 = DownTransition(16, 1, elu, dropout = True, attention_module = attention_module)
        self.down_tr64 = DownTransition(32, 2, elu, dropout=True, attention_module = attention_module)
        self.down_tr128 = DownTransition(64, 2, elu, dropout=True, attention_module = attention_module)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True, attention_module = attention_module)
        self.up_tr256 = UpTransition(256, 256, 1, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 1, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu, dropout = True)
        self.up_tr32 = UpTransition(64, 32, 1, elu, dropout = True)
        self.out_tr = OutputTransition(32, classes, elu)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out

    def test(self,device='cpu'):
        input_tensor = torch.rand(1, self.in_channels, 32, 32, 32)
        ideal_out = torch.rand(1, self.classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self.to(torch.device(device)), (self.in_channels, 32, 32, 32),device=device)
        # import torchsummaryX
        # torchsummaryX.summary(self, input_tensor.to(device))
        print("Vnet test is complete")
"""


#m = VNetAttention(in_channels=1,classes=1)
#m.test()


class VNetAttention(nn.Module):
    def __init__(self, elu=False, in_channels=1, classes=1, attention_module = True):
        super(VNetAttention, self).__init__()
        self.classes = classes
        self.in_channels = in_channels

        self.in_tr = InputTransition(in_channels, elu=elu, attention_module = attention_module)
        self.down_tr16 = DownTransition(8, 1, elu, attention_module = attention_module)
        self.down_tr32 = DownTransition(16, 1, elu, attention_module = attention_module)
        self.down_tr64 = DownTransition(32, 2, elu, dropout=False, attention_module = attention_module)
        self.down_tr128 = DownTransition(64, 2, elu, dropout=False, attention_module = attention_module)
        #self.up_tr256 = UpTransition(256, 256, 1, elu, dropout=True)
        self.up_tr128 = UpTransition(128, 128, 1, elu, dropout=False)
        self.up_tr64 = UpTransition(128, 64, 1, elu, dropout = False)
        self.up_tr32 = UpTransition(64, 32, 1, elu, dropout=False)
        self.up_tr16 = UpTransition(32, 16, 1, elu, dropout = False)
        self.out_tr = OutputTransition(16, classes, elu)

    def forward(self, x):
        out8 = self.in_tr(x)
        out16 = self.down_tr16(out8)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out = self.up_tr128(out128, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.up_tr16(out, out8)
        out = self.out_tr(out)
        return out

    def test(self,device='cpu'):
        input_tensor = torch.rand(1, self.in_channels, 32, 32, 32)
        ideal_out = torch.rand(1, self.classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self.to(torch.device(device)), (self.in_channels, 32, 32, 32),device=device)
        print("Vnet Attention test is complete!")


m = VNetAttention(in_channels=1, classes=1)
m.test()
