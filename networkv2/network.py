import numpy as np
import torch
from torch import nn
from torch import autograd
import functools
from networkv2.blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock, ResidualOctBlock_basic, ActFirst_no_normalization, oct_conv7x7, norm_conv7x7,oct_conv5x5,oct_conv4x4, norm_conv4x4, oct_conv3x3, norm_conv3x3,oct_conv1x1,norm_conv1x1, Oct_conv_norm, conv_norm, Oct_conv_reLU, Oct_conv_lreLU, Oct_conv_up
#OctConv Generator
class ContentEncoder(nn.Module):
    def __init__(self,input_dim, dim, norm):
        super(ContentEncoder, self).__init__()
        norm=='in'
        oct_conv_on = True
        alpha_in, alpha_out = 0.5, 0.5
        conv7x7 = oct_conv7x7 if oct_conv_on else norm_conv7x7
        conv4x4 = oct_conv4x4 if oct_conv_on else norm_conv4x4
        conv3x3 = oct_conv3x3 if oct_conv_on else norm_conv3x3
        norm_func = Oct_conv_norm if oct_conv_on else conv_norm
        act_func = Oct_conv_reLU if oct_conv_on else nn.ReLU

        # first layer:keep same resolution
        # both of alpha_in and  alpha_out does not matter 
        self.conv1 = conv7x7(in_planes=input_dim, out_planes=dim, alpha_in=alpha_in, alpha_out=alpha_out, padding=3, type="first")
        self.bn1 = norm_func(planes = dim, norm = norm)
        self.re1 = act_func(inplace=False)

        # second layer:redcue resolution hafives time
        self.conv2 = conv4x4(in_planes=dim, out_planes=dim*2, alpha_in=alpha_in, alpha_out=alpha_out, padding=1, type="normal")
        self.bn2 = norm_func(planes = dim*2, norm = norm)
        self.re2 = act_func(inplace=False)

        # three layer:redcue resolution hafives time
        self.conv3 = conv4x4(in_planes=dim*2, out_planes=dim*4, alpha_in=alpha_in, alpha_out=alpha_out, padding=1, type="normal")
        self.bn3 = norm_func(planes = dim*4, norm = norm)
        self.re3 = act_func(inplace=False)
        # residual blocks
        self.Res1 = ResidualOctBlock_basic(conv_dim=dim*4, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=norm)
        self.Res2 = ResidualOctBlock_basic(conv_dim=dim*4, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=norm)
        self.Res3 = ResidualOctBlock_basic(conv_dim=dim*4, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=norm)
        self.Res4 = ResidualOctBlock_basic(conv_dim=dim*4, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=norm)
        self.Res5 = ResidualOctBlock_basic(conv_dim=dim*4, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=norm)
        self.Res6 = ResidualOctBlock_basic(conv_dim=dim*4, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=norm)

       # self.model = []
       # self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
       # # downsampling blocks
       # for i in range(n_downsample):
       #     self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
       #     dim *= 2
        # residual blocks
       # self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
       # self.model = nn.Sequential(*self.model)
        self.output_dim = dim

   # def forward(self, x):
   #     return self.model(x)
    def forward(self, x, alpha_in, alpha_out):
        # first layer:keep same resolution
        out=self.conv1(x, alpha_in=alpha_in, alpha_out=alpha_out)
        out=self.bn1(out)
        out=self.re1(out)
        # second layer:redcue resolution hafives time
        out=self.conv2(out, alpha_in=alpha_in, alpha_out=alpha_out)
        out=self.bn2(out)
        out=self.re2(out)
        # three layer:redcue resolution hafives time
        out=self.conv3(out, alpha_in=alpha_in, alpha_out=alpha_out)
        out=self.bn3(out)
        out=self.re3(out)
        # residual blocks


        out = self.Res1(out, alpha_in, alpha_out)
        out = self.Res2(out, alpha_in, alpha_out)
        out = self.Res3(out, alpha_in, alpha_out)
        out = self.Res4(out, alpha_in, alpha_out)
        out = self.Res5(out, alpha_in, alpha_out)
        out = self.Res6(out, alpha_in, alpha_out)

        return out

class Decoder(nn.Module):
    def __init__(self,dim, out_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()
        oct_conv_on = True
        norm='in'
        alpha_in, alpha_out = 0.5, 0.5
        conv7x7 = oct_conv7x7 if oct_conv_on else norm_conv7x7
        #conv5x5 = oct_conv5x5 if oct_conv_on else norm_conv5x5
        conv4x4 = oct_conv4x4 if oct_conv_on else norm_conv4x4
        conv3x3 = oct_conv3x3 if oct_conv_on else norm_conv3x3
        norm_func = Oct_conv_norm if oct_conv_on else conv_norm
        act_func = Oct_conv_reLU if oct_conv_on else nn.ReLU
        up_func = Oct_conv_up if oct_conv_on else nn.Upsample 

        # residual blocks
        self.Res1 = ResidualOctBlock_basic(conv_dim=4*dim, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=res_norm)
        self.Res2 = ResidualOctBlock_basic(conv_dim=4*dim, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=res_norm)
        self.Res3 = ResidualOctBlock_basic(conv_dim=4*dim, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=res_norm)
        self.Res4 = ResidualOctBlock_basic(conv_dim=4*dim, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=res_norm)
        self.Res5 = ResidualOctBlock_basic(conv_dim=4*dim, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=res_norm)
        self.Res6 = ResidualOctBlock_basic(conv_dim=4*dim, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=res_norm)

        # first layer:double resolution
        dim = 4*dim
        self.up1 = up_func(scale_factor=2)
        self.conv1 = conv3x3(in_planes=dim, out_planes=dim//2, alpha_in=alpha_in, alpha_out=alpha_out, padding=1, type="normal")
        self.bn1 = norm_func(planes = dim//2, norm = norm)
        self.re1 = act_func(inplace=True)

        # second layer:double resolution 
        self.up2 = up_func(scale_factor=2)
        self.conv2 = conv3x3(in_planes=dim//2, out_planes=dim//4, alpha_in=alpha_in, alpha_out=alpha_out, padding=1,  type="normal")
        self.bn2 = norm_func(planes = dim//4, norm = norm)
        self.re2 = act_func(inplace=False)
        # three layer:keep same resolution
        self.conv3 = conv7x7(in_planes=dim//4, out_planes=out_dim, alpha_in=alpha_in, alpha_out=alpha_out, padding=3, type="last")
        self.tanh = nn.Tanh()

    def forward(self, x, alpha_in, alpha_out):

        # residual blocks
        out = self.Res1(x, alpha_in, alpha_out)
        out = self.Res2(out, alpha_in, alpha_out)
        out = self.Res3(out, alpha_in, alpha_out)
        out = self.Res4(out, alpha_in, alpha_out)
        out = self.Res5(out, alpha_in, alpha_out)
        out = self.Res6(out, alpha_in, alpha_out)

        # first layer:double resolution
        out=self.up1(out)
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


class GPPatchMcResDis(nn.Module):
    def __init__(self,n_res_blks, nf):
        super(GPPatchMcResDis, self).__init__()
        assert n_res_blks % 2 == 0, 'n_res_blk must be multiples of 2'
        self.n_layers = n_res_blks // 2
        nf = nf
        cnn_f = [Conv2dBlock(3, nf, 7, 1, 3,
                             pad_type='reflect',
                             norm='none',
                             activation='none')]
        for i in range(self.n_layers - 1):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])
        nf_out = 1
        cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
        cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
        # cnn_c = [Conv2dBlock(nf_out, hp['num_classes'], 1, 1,
        #                      norm='none',
        #                      activation='lrelu',
        #                      activation_first=True)]
        self.cnn_f = nn.Sequential(*cnn_f)
        #self.cnn_c = nn.Sequential(*cnn_c)

    def forward(self, x, alpha_in=0., alpha_out=0.):
        #assert(x.size(0) == y.size(0))
        feat = self.cnn_f(x)
        return feat

    def calc_dis_fake_loss(self, input_fake, input_label,octave_alpha):
        resp_fake, gan_feat = self.forward(input_fake, input_label)
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).cuda()
        fake_loss = torch.nn.ReLU()(1.0 + resp_fake).mean()
        correct_count = (resp_fake < 0).sum()
        fake_accuracy = correct_count.type_as(fake_loss) / total_count
        return fake_loss, fake_accuracy, resp_fake

    def calc_dis_real_loss(self, input_real, input_label,octave_alpha):
        resp_real, gan_feat = self.forward(input_real, input_label)
        total_count = torch.tensor(np.prod(resp_real.size()),
                                   dtype=torch.float).cuda()
        real_loss = torch.nn.ReLU()(1.0 - resp_real).mean()
        correct_count = (resp_real >= 0).sum()
        real_accuracy = correct_count.type_as(real_loss) / total_count
        return real_loss, real_accuracy, resp_real

    def calc_gen_loss(self, input_fake, input_fake_label,octave_alpha):
        resp_fake, gan_feat = self.forward(input_fake, input_fake_label)
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).cuda()
        loss = -resp_fake.mean()
        correct_count = (resp_fake >= 0).sum()
        accuracy = correct_count.type_as(loss) / total_count
        return loss, accuracy, gan_feat

    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum()/batch_size
        return reg

class GPPatchMcResDis_yaxing(nn.Module):
    def __init__(self,n_res_blks, nf):
        super(GPPatchMcResDis, self).__init__()
        assert n_res_blks % 2 == 0, 'n_res_blk must be multiples of 2'
        self.n_layers = n_res_blks // 2
        nf = nf
        #cnn_f = [Conv2dBlock(3, nf, 7, 1, 3,
        #                     pad_type='reflect',
        #                     norm='none',
        #                     activation='none')]
        oct_conv_on = True
        conv7x7 = oct_conv7x7 if oct_conv_on else norm_conv7x7
        conv4x4 = oct_conv4x4 if oct_conv_on else norm_conv4x4
        conv3x3 = oct_conv3x3 if oct_conv_on else norm_conv3x3
        conv1x1 = oct_conv1x1 if oct_conv_on else norm_conv1x1
        norm_func = Oct_conv_norm if oct_conv_on else conv_norm
        act_func = Oct_conv_reLU if oct_conv_on else nn.ReLU
        avgpool = Oct_avgpool if oct_conv_on else nn.AvgPool2d
        reflePad = Oct_reflectionPad if oct_conv_on else nn.ReflectionPad2d()

        # initialize padding: for input image
        self.pad1 = nn.ReflectionPad2d(padding=3)
        # first layer Conv-64, original paper only use conv
        self.conv1 = conv7x7(in_planes=3, out_planes=nf, padding=3, type="first")

        # loop1 
        # first resblock128
        # second resblock128: firstly conduct conv1*1, then 2 conv3*3, final avgpool
        # conv1*1:padding is 0, note: original code is ActFirst which firstly perform conv1*1, then 2 conv3*3
        self.res1 = ActFirst_no_normalization(conv_dim=nf, oct_conv_on = oct_conv_on, norm='none')
        self.conv2 = conv1x1(in_planes=nf, out_planes=nf*2, padding=0, type="normal")
        self.res2 = ActFirst_no_normalization(conv_dim=nf*2, oct_conv_on = oct_conv_on, norm='none')
        self.pad2 = reflePad(padding=1)
        self.avgp1 = avgpool(kernel_size=3, stride=2)

        # loop2 
        self.res3 = ActFirst_no_normalization(conv_dim=nf*2, oct_conv_on = oct_conv_on, norm='none')
        self.conv3 = conv1x1(in_planes=nf*2, out_planes=nf*4, padding=0, type="normal")
        self.res4 = ActFirst_no_normalization(conv_dim=nf*4, oct_conv_on = oct_conv_on, norm='none')
        self.pad3 = reflePad(padding=1)
        self.avgp2 = avgpool(kernel_size=3, stride=2)

        # loop3 
        self.res5 = ActFirst_no_normalization(conv_dim=nf*4, oct_conv_on = oct_conv_on, norm='none')
        self.conv4 = conv1x1(in_planes=nf*4, out_planes=nf*8, padding=0, type="normal")
        self.res6 = ActFirst_no_normalization(conv_dim=nf*8, oct_conv_on = oct_conv_on, norm='none')
        self.pad4 = reflePad(padding=1)
        self.avgp3 = avgpool(kernel_size=3, stride=2)
        # loop4 
        self.res7 = ActFirst_no_normalization(conv_dim=nf*8, oct_conv_on = oct_conv_on, norm='none')
        self.conv5 = conv1x1(in_planes=nf*8, out_planes=nf*16, padding=0, type="normal")
        self.res8 = ActFirst_no_normalization(conv_dim=nf*16, oct_conv_on = oct_conv_on, norm='none')
        self.pad5 = reflePad(padding=1)
        self.avgp4 = avgpool(kernel_size=3, stride=2)
        # final block 
        self.res9 = ActFirst_no_normalization(conv_dim=nf*16, oct_conv_on = oct_conv_on, norm='none')
        self.res10 = ActFirst_no_normalization(conv_dim=nf*16, oct_conv_on = oct_conv_on, norm='none')
        # extract layer: to merget tow braches 
        self.leakre1 =Oct_conv_lreLU(negative_slope=0.2, inplace=False)
        self.cnn_f = conv1x1(in_planes=nf*16, out_planes=nf*16, padding=0, type="last")
        self.leakre2 = nn.LeakyReLU(0.2, inplace=False)
        self.cnn_c = norm_conv1x1(in_planes=nf*16, out_planes=hp['num_classes'], padding=0)

    def forward(self, x, y, alpha_in, alpha_out):
        assert(x.size(0) == y.size(0))
        # feature
        #output = self.pad1(x)
        output = self.conv1(x, alpha_in=alpha_in, alpha_out=alpha_out)
        # loop1 
        output = self.res1(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.conv2(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.res2(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.pad2(output) 
        output = self.avgp1(output) 

        # loop2 
        output = self.res3(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.conv3(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.res4(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.pad3(output) 
        output = self.avgp2(output) 

        # loop3 
        output = self.res5(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.conv4(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.res6(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.pad4(output) 
        output = self.avgp3(output) 
        # loop4 
        output = self.res7(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.conv5(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.res8(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.pad5(output) 
        output = self.avgp4(output) 
        # final block 
        output = self.res9(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.res10(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        # extract layer: to merget tow braches 
        output = self.leakre1(output)
        feat = self.cnn_f(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        feat1 = self.leakre2(feat)
        out = self.cnn_c(feat1)
        index = torch.LongTensor(range(out.size(0))).cuda()
        out = out[index, y, :, :]

       # feat = self.cnn_f(x)
       # out = self.cnn_c(feat)
       # index = torch.LongTensor(range(out.size(0))).cuda()
       # out = out[index, y, :, :]
        return out, feat
    def calc_dis_fake_loss(self, input_fake, input_label, octave_alpha):
        resp_fake, gan_feat = self.forward(input_fake, input_label, alpha_in=octave_alpha, alpha_out=octave_alpha)
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).cuda()
        fake_loss = torch.nn.ReLU()(1.0 + resp_fake).mean()
        correct_count = (resp_fake < 0).sum()
        fake_accuracy = correct_count.type_as(fake_loss) / total_count
        return fake_loss, fake_accuracy, resp_fake

    def calc_dis_real_loss(self, input_real, input_label, octave_alpha):
        resp_real, gan_feat = self.forward(input_real, input_label, alpha_in=octave_alpha, alpha_out=octave_alpha)
        total_count = torch.tensor(np.prod(resp_real.size()),
                                   dtype=torch.float).cuda()
        real_loss = torch.nn.ReLU()(1.0 - resp_real).mean()
        correct_count = (resp_real >= 0).sum()
        real_accuracy = correct_count.type_as(real_loss) / total_count
        return real_loss, real_accuracy, resp_real

    def calc_gen_loss(self, input_fake, input_fake_label, octave_alpha):
        resp_fake, gan_feat = self.forward(input_fake, input_fake_label, alpha_in=octave_alpha, alpha_out=octave_alpha)
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).cuda()
        loss = -resp_fake.mean()
        correct_count = (resp_fake >= 0).sum()
        accuracy = correct_count.type_as(loss) / total_count
        return loss, accuracy, gan_feat

    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum()/batch_size
        return reg

# Projection of x onto y
def proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())

# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
  for y in ys:
    x = x - proj(x, y)
  return x

import torch.nn.functional as F
# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
  # Lists holding singular vectors and values
  us, vs, svs = [], [], []
  for i, u in enumerate(u_):
    # Run one step of the power iteration
    with torch.no_grad():
      v = torch.matmul(u, W)
      # Run Gram-Schmidt to subtract components of all other singular vectors
      v = F.normalize(gram_schmidt(v, vs), eps=eps)
      # Add to the list
      vs += [v]
      # Update the other singular vector
      u = torch.matmul(v, W.t())
      # Run Gram-Schmidt to subtract components of all other singular vectors
      u = F.normalize(gram_schmidt(u, us), eps=eps)
      # Add to the list
      us += [u]
      if update:
        u_[i][:] = u
    # Compute this singular value and add it to the list
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
  return svs, us, vs

class SN(object):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    # Number of power iterations per step
    self.num_itrs = num_itrs
    # Number of singular values
    self.num_svs = num_svs
    # Transposed?
    self.transpose = transpose
    # Epsilon value for avoiding divide-by-0
    self.eps = eps
    # Register a singular vector for each sv
    for i in range(self.num_svs):
      self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
      self.register_buffer('sv%d' % i, torch.ones(1))
  
  # Singular vectors (u side)
  @property
  def u(self):
    return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

  # Singular values; 
  # note that these buffers are just for logging and are not used in training. 
  @property
  def sv(self):
   return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]
   
  # Compute the spectrally-normalized weight
  def W_(self):
    W_mat = self.weight.view(self.weight.size(0), -1)
    if self.transpose:
      W_mat = W_mat.t()
    # Apply num_itrs power iterations
    for _ in range(self.num_itrs):
      svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps) 
    # Update the svs
    if self.training:
      with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
        for i, sv in enumerate(svs):
          self.sv[i][:] = sv     
    return self.weight / svs[0]

class SNConv2d(nn.Conv2d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True, 
             num_svs=1, num_itrs=1, eps=1e-12):
    nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, 
                     padding, dilation, groups, bias)
    SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)    
  def forward(self, x):
    return F.conv2d(x, self.W_(), self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)

class SNLinear(nn.Linear, SN):
  def __init__(self, in_features, out_features, bias=True,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Linear.__init__(self, in_features, out_features, bias)
    SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
  def forward(self, x):
    return F.linear(x, self.W_(), self.bias)

class SNEmbedding(nn.Embedding, SN):
  def __init__(self, num_embeddings, embedding_dim, padding_idx=None, 
               max_norm=None, norm_type=2, scale_grad_by_freq=False,
               sparse=False, _weight=None,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx,
                          max_norm, norm_type, scale_grad_by_freq, 
                          sparse, _weight)
    SN.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)
  def forward(self, x):
    return F.embedding(x, self.W_())

# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
from torch.nn import Parameter as P
class Attention(nn.Module):
  def __init__(self, ch, which_conv=SNConv2d, name='attention'):
    super(Attention, self).__init__()
    # Channel multiplier
    self.ch = ch
    self.which_conv = which_conv
    self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
    self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
    # Learnable gain parameter
    self.gamma = P(torch.tensor(0.), requires_grad=True)
  def forward(self, x, y=None):
    # Apply convs
    theta = self.theta(x)
    phi = F.max_pool2d(self.phi(x), [2,2])
    g = F.max_pool2d(self.g(x), [2,2])    
    # Perform reshapes
    theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
    phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
    g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
    return self.gamma * o + x

class DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
               preactivation=True, activation=None, downsample=None,
               channel_ratio=4):
    super(DBlock, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
    self.hidden_channels = self.out_channels // channel_ratio
    self.which_conv = which_conv
    self.preactivation = preactivation
    self.activation = activation
    self.downsample = downsample
        
    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.hidden_channels, 
                                 kernel_size=1, padding=0)
    self.conv2 = self.which_conv(self.hidden_channels, self.hidden_channels)
    self.conv3 = self.which_conv(self.hidden_channels, self.hidden_channels)
    self.conv4 = self.which_conv(self.hidden_channels, self.out_channels, 
                                 kernel_size=1, padding=0)
                                 
    self.learnable_sc = True if (in_channels != out_channels) else False
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels - in_channels, 
                                     kernel_size=1, padding=0)
  def shortcut(self, x):
    if self.downsample:
      x = self.downsample(x)
    if self.learnable_sc:
      x = torch.cat([x, self.conv_sc(x)], 1)    
    return x
  def forward(self, x):
    # 1x1 bottleneck conv
    h = self.conv1(F.relu(x))
    # 3x3 convs
    h = self.conv2(self.activation(h))
    h = self.conv3(self.activation(h))
    # relu before downsample
    h = self.activation(h)
    # downsample
    if self.downsample:
      h = self.downsample(h)     
    # final 1x1 conv
    h = self.conv4(h)
    return h + self.shortcut(x)


def D_arch(ch=64, attention='32',ksize='333333', dilation='111111'):
  arch = {}

  arch[64]  = {'in_channels' :  [item * ch for item in [1, 2]],
               'out_channels' : [item * ch for item in [2, 4]],
               'downsample' : [True] * 2 + [False],
               'resolution' : [32, 16],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,7)}}

  return arch


import torch.optim as optim
from torch.nn import init
class BigGanDiscriminator(nn.Module):
    def __init__(self, in_nc=3, out_nc=1, D_wide=True, D_depth=2, resolution=64,
               D_kernel_size=3, D_attn='64', n_classes=1000,
               num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
               D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
               SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
               D_init='ortho', skip_init=False, D_param='SN', **kwargs):
        super(BigGanDiscriminator, self).__init__()
        # Width multiplier
        self.ch = in_nc
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # How many resblocks per stage?
        self.D_depth = D_depth
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16
        # Architecture
        self.arch = D_arch(self.ch, self.attention)[resolution]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = functools.partial(SNConv2d,
                                kernel_size=3, padding=1,
                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                eps=self.SN_eps)
            self.which_linear = functools.partial(SNLinear,
                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                eps=self.SN_eps)
            self.which_embedding = functools.partial(SNEmbedding,
                                    num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                    eps=self.SN_eps)

        # Prepare model
        # Stem convolution
        self.input_conv = self.which_conv(3, self.arch['in_channels'][0])
        self.norm = nn.InstanceNorm2d(self.ch * 8)
        self.sigmoid = nn.Sigmoid()
        self.last_conv2d = nn.Conv2d(12, 1, 3, 1, 1)
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index] if d_index==0 else self.arch['out_channels'][index],
                            out_channels=self.arch['out_channels'][index],
                            which_conv=self.which_conv,
                            wide=self.D_wide,
                            activation=self.activation,
                            preactivation=True,
                            downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] and d_index==0 else None))
                            for d_index in range(self.D_depth)]]
        # If attention on this block, attach it to the end
        if self.arch['attention'][self.arch['resolution'][index]]:
            print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
            self.blocks[-1] += [Attention(self.arch['out_channels'][index],
                                                self.which_conv)]
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
        # Embedding for projection discrimination
        self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

        # Initialize weights
        if not skip_init:
            self.init_weights()

        # Set up optimizer
        self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
        if D_mixed_precision:
            print('Using fp16 adam in D...')
            import utils
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' % self.param_count)

    def forward(self, x, y=None):
        # Run input conv
        h = self.input_conv(x)
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        # Apply global sum pooling as in SN-GAN'
        h = self.norm(h)
        # out = torch.sum(self.activation(h), [1], keepdim=True) -> 32 1 16 344
        m = self.last_conv2d(h)
        out = self.sigmoid(m)
        return out 
        
    