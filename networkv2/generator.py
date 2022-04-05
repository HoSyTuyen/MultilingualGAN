
from matplotlib.pyplot import cla
from torch.nn.modules import conv

import utils
import torch.nn.functional as F
import torch
from torch import nn
from networkv2.network import ContentEncoder, Decoder
from networkv2.blocks import oct_conv7x7,oct_conv3x3,Oct_conv_up,OctConv,Oct_conv_norm, Oct_conv_reLU, norm_conv3x3, conv_norm
class GeneratorOctConv(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=32, nb=4):
        super(GeneratorOctConv, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.nb = nb

        self.enc_content = ContentEncoder(3,
                                          nf,
                                          'in')

        self.dec = Decoder(self.enc_content.output_dim,
                           3,
                           res_norm='adain',
                           activ='relu',
                           pad_type='reflect')
    def forward(self, x):
        out = self.enc_content(x,0.5,0.5)
        out = self.dec(out,0.5,0.5)
        return out

########################################################################################################
########################################     BASELINE OCTCONV     ######################################
class ResidualOctBlock_basic(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, conv_dim=64, alpha_in=0.25, alpha_out=0.25, oct_conv_on = True,  norm='in'):
        super(ResidualOctBlock_basic, self).__init__()
        oct_conv_on = True
        alpha_in, alpha_out = 0.5, 0.5
        conv3x3 = oct_conv3x3 if oct_conv_on else norm_conv3x3
        norm_func = Oct_conv_norm if oct_conv_on else conv_norm
        act_func = Oct_conv_reLU if oct_conv_on else nn.ReLU

        self.conv1 = conv3x3(in_planes=conv_dim, out_planes=conv_dim, alpha_in=alpha_in, alpha_out=alpha_out, padding=1, type="normal")
        self.bn1 = norm_func(planes = conv_dim, norm = norm)
        self.re1 = act_func(inplace=True)

        self.conv2 = conv3x3(in_planes=conv_dim, out_planes=conv_dim, alpha_in=alpha_in, alpha_out=alpha_out, padding=1, type="normal")
        self.bn2 = norm_func(planes = conv_dim, norm = norm)

    def forward(self, x, alpha_in=0.25, alpha_out=0.25):
        #basic block: 2 conv + input
        #conv1
        out=self.conv1(x, alpha_in=alpha_in, alpha_out=alpha_out)
        out=self.bn1(out)
        out=self.re1(out)
        #conv2
        out=self.conv2(out, alpha_in=alpha_in, alpha_out=alpha_out)
        out=self.bn2(out)
        return x[0] + out[0], x[1] + out[1]
        
class resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding,alpha_in,alpha_out):
        super(resnet_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        #self.conv1 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.conv1 = OctConv(channel, channel, kernel, stride, padding,alpha_in=alpha_in,alpha_out=alpha_out, type='normal')
        #self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv1_norm = Oct_conv_norm(channel,alpha_in=alpha_in,alpha_out=alpha_out)
        #self.conv2 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.re1 = Oct_conv_reLU()
        self.conv2 = OctConv(channel, channel, kernel, stride, padding,alpha_in=alpha_in,alpha_out=alpha_out, type='normal')
        #self.conv2_norm = nn.InstanceNorm2d(channel)
        self.conv2_norm = Oct_conv_norm(channel,alpha_in=alpha_in,alpha_out=alpha_out)

        utils.initialize_weights(self)

    def forward(self, input, alpha_in = 0.5, alpha_out=0.5):
        # x = F.relu(self.conv1_norm(self.conv1(input)), True)
        # x = self.conv2_norm(self.conv2(x))
        #Conv1
        out = self.conv1(input, alpha_in=alpha_in, alpha_out=alpha_out)
        out = self.conv1_norm(out, alpha_in=alpha_in, alpha_out=alpha_out)
        out = self.re1(out)
        #Conv2
        out = self.conv2(out, alpha_in=alpha_in, alpha_out=alpha_out)
        out = self.conv2_norm(out, alpha_in=alpha_in, alpha_out=alpha_out)

        return input[0] + out[0], input[1] + out[1] 

class BaselineOctConvDown(nn.Module):
    def __init__(self, in_nc=3, nf=32, alpha_in=0.5, alpha_out=0.5):
        super(BaselineOctConvDown, self).__init__()
        self.input_nc = in_nc
        self.nf = nf
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.OctConv1 = OctConv(in_nc, nf, 7, 1, 3,alpha_in=0.5,alpha_out=0.5, type='first')
        #self.IN1 = nn.InstanceNorm2d(nf),
        self.IN1 = Oct_conv_norm(nf,alpha_in=0.5,alpha_out=0.5)
        self.ReLU = Oct_conv_reLU(inplace=True)
        self.OctConv2 = OctConv(nf, nf * 2, 3, 2, 1,alpha_in=0.5,alpha_out=0.5, type='normal')
        self.OctConv3 = OctConv(nf * 2, nf * 2, 3, 1, 1,alpha_in=0.5,alpha_out=0.5, type='normal'),
        #self.OctConv2 = OctConv(nf, nf * 2, 3, 2, 1,alpha_in=0.5,alpha_out=0.5, type='normal')
        #self.OctConv3 = OctConv(nf * 2, nf * 2, 3, 2, 1,alpha_in=0.5,alpha_out=0.5, type='normal')
        
        #self.IN2 = nn.InstanceNorm2d(nf * 2),
        self.IN2 = Oct_conv_norm(nf * 2,alpha_in=0.5,alpha_out=0.5)
        self.OctConv4 = OctConv(nf * 2, nf * 4, 3, 2, 1,alpha_in=0.5,alpha_out=0.5, type='normal')
        self.OctConv5 = OctConv(nf * 4, nf * 4, 3, 1, 1,alpha_in=0.5,alpha_out=0.5, type='normal')
        self.IN3 = Oct_conv_norm(nf * 4,alpha_in=0.5,alpha_out=0.5)

    def forward(self, x):
        out = self.OctConv1(x,self.alpha_in,self.alpha_out)
        out = self.IN1(out)
        out = self.ReLU(out)

        out = self.OctConv2(out,self.alpha_in,self.alpha_out)
        #out = self.OctConv3(out,self.alpha_in,self.alpha_out)
        out = self.IN2(out)
        out = self.ReLU(out)

        out = self.OctConv4(out,self.alpha_in,self.alpha_out)
        #out = self.OctConv5(out,self.alpha_in,self.alpha_out)
        out = self.IN3(out)
        out = self.ReLU(out)
        return out

class BaselineOctConvUp(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=32, alpha_in=0.5, alpha_out=0.5):
        super(BaselineOctConvUp,self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        
        # self.OctConvUp1 = Oct_conv_up(scale_factor=2)
        # self.OctCov1 = OctConv(nf * 4, nf * 2, 3, 1, 1,alpha_in=0.5,alpha_out=0.5,type="normal")
        # self.IN1 = Oct_conv_norm(nf * 2,alpha_in=0.5,alpha_out=0.5)
        # self.re1 = Oct_conv_reLU(inplace=True)
        
        # self.OctConvUp2 = Oct_conv_up(scale_factor=2)
        # self.OctConv2 = OctConv(nf * 2, nf, 3, 1, 1,alpha_in=0.5,alpha_out=0.5,type="last"),
        # self.IN2 = nn.InstanceNorm2d(nf)
        # self.re2 = nn.ReLU(True)
        # self.Tanh = nn.Tanh()
        # first layer:double resolution
        conv7x7 = oct_conv7x7
        #conv5x5 = oct_conv5x5 if oct_conv_on else norm_conv5x5
        conv3x3 = oct_conv3x3
        norm_func = Oct_conv_norm
        act_func = Oct_conv_reLU
        up_func = Oct_conv_up
        dim = 4*nf
        self.up1 = up_func(scale_factor=2)
        self.conv1 = conv3x3(in_planes=dim, out_planes=dim//2, alpha_in=alpha_in, alpha_out=alpha_out, padding=1, type="normal")
        self.bn1 = norm_func(planes = dim//2, norm = 'in')
        self.re1 = act_func(inplace=True)

        # second layer:double resolution 
        self.up2 = up_func(scale_factor=2)
        self.conv2 = conv3x3(in_planes=dim//2, out_planes=dim//4, alpha_in=alpha_in, alpha_out=alpha_out, padding=1,  type="normal")
        self.bn2 = norm_func(planes = dim//4, norm = 'in')
        self.re2 = act_func(inplace=False)
        # three layer:keep same resolution
        self.conv3 = conv7x7(in_planes=dim//4, out_planes=3, alpha_in=alpha_in, alpha_out=alpha_out, padding=3, type="last")
        self.tanh = nn.Tanh()

    def forward(self, x, alpha_in = 0.5, alpha_out=0.5):
        # out = self.OctConvUp1(x)
        # out = self.OctCov1(out,self.alpha_in,self.alpha_out)
        # out = self.IN1(out)
        # out = self.re1(out)

        # out = self.OctConvUp2(out)
        # out = self.OctConv2(out,self.alpha_in,self.alpha_out)
        # out = self.IN2(out)
        # out = self.re2(out)
        # out = self.Tanh(out)
        # return out

        # first layer:double resolution
        out=self.up1(x)
        out=self.conv1(out, alpha_in, alpha_out)
        out=self.bn1(out)
        out=self.re1(out)
        # second layer:double resolution hafives time
        out=self.up2(out)
        out=self.conv2(out, alpha_in, alpha_out)
        out=self.bn2(out)
        out=self.re2(out)
        # three layer:keep same resolution 
        out=self.conv3(out, alpha_in, alpha_out)
        out=self.tanh(out)

        return out

class GeneratorBaselineOctConv(nn.Module):
    # initializers
    def __init__(self, in_nc=3, out_nc=3, nf=32, nb=4):
        super(GeneratorBaselineOctConv, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.nb = nb
        # self.down_convs = nn.Sequential(
        #     #nn.Conv2d(in_nc, nf, 7, 1, 3),
        #     OctConv(in_nc, nf, 7, 1, 3,alpha_in=0.5,alpha_out=0.5),
        #     nn.InstanceNorm2d(nf),
        #     nn.ReLU(True),
        #     #nn.Conv2d(nf, nf * 2, 3, 2, 1),
        #     OctConv(nf, nf * 2, 3, 2, 1,alpha_in=0.5,alpha_out=0.5), 
        #     #nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
        #     OctConv(nf * 2, nf * 2, 3, 1, 1,alpha_in=0.5,alpha_out=0.5),
        #     nn.InstanceNorm2d(nf * 2),
        #     nn.ReLU(True),
        #     #nn.Conv2d(nf * 2, nf * 4, 3, 2, 1),
        #     #nn.Conv2d(nf * 4, nf * 4, 3, 1, 1), 
        #     nn.InstanceNorm2d(nf * 4),
        #     nn.ReLU(True),
        # )
        self.down_convs = BaselineOctConvDown(in_nc, nf, alpha_in=0.5, alpha_out=0.5)

        #self.resnet_blocks = []
        #for i in range(nb):
        #    self.resnet_blocks.append(resnet_block(nf * 4, 3, 1, 1,alpha_in=0.5,alpha_out=0.5))
        #self.resnet_blocks = nn.Sequential(*self.resnet_blocks)
        self.Res1 = ResidualOctBlock_basic(conv_dim=nf*4, alpha_in=0.5, alpha_out=0.5, oct_conv_on = True, norm='in')
        self.Res2 = ResidualOctBlock_basic(conv_dim=nf*4, alpha_in=0.5, alpha_out=0.5, oct_conv_on = True, norm='in')
        self.Res3 = ResidualOctBlock_basic(conv_dim=nf*4, alpha_in=0.5, alpha_out=0.5, oct_conv_on = True, norm='in')
        self.Res4 = ResidualOctBlock_basic(conv_dim=nf*4, alpha_in=0.5, alpha_out=0.5, oct_conv_on = True, norm='in')
        # self.up_convs = nn.Sequential(
        #     #nn.ConvTranspose2d(nf * 4, nf * 2, 3, 2, 1, 1),
        #     Oct_conv_up(),
        #     #nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
        #     OctConv(nf * 2, nf * 2, 3, 1, 1,alpha_in=0.5,alpha_out=0.5),
        #     nn.InstanceNorm2d(nf * 2),
        #     nn.ReLU(True),
        #     #nn.ConvTranspose2d(nf * 2, nf, 3, 2, 1, 1),
        #     Oct_conv_up(),
        #     #nn.Conv2d(nf, nf, 3, 1, 1),
        #     OctConv(nf, nf, 3, 1, 1,alpha_in=0.5,alpha_out=0.5),
        #     nn.InstanceNorm2d(nf),
        #     nn.ReLU(True),
        #     #nn.Conv2d(nf, out_nc, 7, 1, 3),
        #     nn.Tanh(),
        # )
        self.up_convs = BaselineOctConvUp(in_nc, out_nc, nf, alpha_in=0.5, alpha_out=0.5)

        utils.initialize_weights(self)

    # forward method
    def forward(self, input):
        x = self.down_convs(input)
        x = self.Res1(x)
        x = self.Res2(x)
        x = self.Res3(x)
        x = self.Res4(x)
        #x = self.resnet_blocks(x)
        output = self.up_convs(x)

        return output