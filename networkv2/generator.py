
from matplotlib.pyplot import cla


import torch
from torch import nn
from networkv2.network import ContentEncoder, Decoder
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


