
from matplotlib.pyplot import cla
from torch.nn.modules import conv

import utils
import torch.nn.functional as F
import torch
from torch import nn
from networkv2.network import ContentEncoder, Decoder
from networkv2.blocks import oct_conv7x7,oct_conv3x3,Oct_conv_up,OctConv
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


class resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding,alpha_in,alpha_out):
        super(resnet_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        #self.conv1 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.conv1 = OctConv(channel, channel, kernel, stride, padding,alpha_in=alpha_in,alpha_out=alpha_out)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        #self.conv2 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.conv2 = OctConv(channel, channel, kernel, stride, padding,alpha_in=alpha_in,alpha_out=alpha_out)
        self.conv2_norm = nn.InstanceNorm2d(channel)

        utils.initialize_weights(self)

    def forward(self, input):
        x = F.relu(self.conv1_norm(self.conv1(input)), True)
        x = self.conv2_norm(self.conv2(x))

        return input + x 

class GeneratorBaselineOctConv(nn.Module):
    # initializers
    def __init__(self, in_nc=3, out_nc=3, nf=32, nb=4):
        super(GeneratorBaselineOctConv, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.nb = nb
        self.down_convs = nn.Sequential(
            #nn.Conv2d(in_nc, nf, 7, 1, 3),
            OctConv(in_nc, nf, 7, 1, 3,alpha_in=0.5,alpha_out=0.5),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            #nn.Conv2d(nf, nf * 2, 3, 2, 1),
            OctConv(nf, nf * 2, 3, 2, 1,alpha_in=0.5,alpha_out=0.5), 
            #nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            OctConv(nf * 2, nf * 2, 3, 1, 1,alpha_in=0.5,alpha_out=0.5),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
            #nn.Conv2d(nf * 2, nf * 4, 3, 2, 1),
            #nn.Conv2d(nf * 4, nf * 4, 3, 1, 1), 
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )

        self.resnet_blocks = []
        for i in range(nb):
            self.resnet_blocks.append(resnet_block(nf * 4, 3, 1, 1,alpha_in=0.5,alpha_out=0.5))

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.up_convs = nn.Sequential(
            #nn.ConvTranspose2d(nf * 4, nf * 2, 3, 2, 1, 1),
            Oct_conv_up(),
            #nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            OctConv(nf * 2, nf * 2, 3, 1, 1,alpha_in=0.5,alpha_out=0.5),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
            #nn.ConvTranspose2d(nf * 2, nf, 3, 2, 1, 1),
            Oct_conv_up,
            #nn.Conv2d(nf, nf, 3, 1, 1),
            OctConv(nf, nf, 3, 1, 1,alpha_in=0.5,alpha_out=0.5),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            #nn.Conv2d(nf, out_nc, 7, 1, 3),
            nn.Tanh(),
        )

        utils.initialize_weights(self)

    # forward method
    def forward(self, input):
        x = self.down_convs(input)
        x = self.resnet_blocks(x)
        output = self.up_convs(x)

        return output