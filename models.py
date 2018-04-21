from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch

import config

##############################################################################################################
# Parameters
##############################################################################################################

img_size = config.img_size
nc = config.nc
nf = config.nf
nz = config.nz


##############################################################################################################
# Functions
##############################################################################################################

def reparametrize(mu, logvar):  # needs to be method to be able to back-propagated -- really?
    std = logvar.mul(0.5).exp_()
    eps = Variable(std.data.new(std.size()).normal_()).cuda()
    return eps.mul(std).add_(mu)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_uniform(m.weight.data, 1.)
        # apply Gaussian weight init? what is standard here?


class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
        self.net = NotImplementedError

    def forward(self, x):
        return self.net(x)


##############################################################################################################
# Custom Layers
##############################################################################################################

class ConvLReluBN(BaseModule):
    def __init__(self, n_in, n_out, kernel_size, stride, padding):
        super(ConvLReluBN, self).__init__()
        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]
        model += [nn.BatchNorm2d(n_out)]
        model += [nn.LeakyReLU(inplace=True)]
        self.net = nn.Sequential(*model)


class UpConvLReluBN(BaseModule):
    def __init__(self, n_in, n_out, kernel_size, stride, padding, mode):
        super(UpConvLReluBN, self).__init__()
        model = []
        model += [nn.Upsample(scale_factor=2, mode=mode)]
        model += [nn.Conv2d(n_in, n_out, kernel_size, stride, padding, bias=False)]
        model += [nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.BatchNorm2d(n_out)]
        self.net = nn.Sequential(*model)


class DeConvLReluBN(BaseModule):
    def __init__(self, n_in, n_out, kernel_size, stride, padding):
        super(DeConvLReluBN, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(n_in, n_out, kernel_size, stride, padding, bias=False)]
        model += [nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.BatchNorm2d(n_out)]
        self.net = nn.Sequential(*model)


##############################################################################################################
# Multiple Layers
##############################################################################################################

class Reducer(BaseModule):
    def __init__(self, n_layer=2, nf=nf):
        super(Reducer, self).__init__()
        model = []
        model += [ConvLReluBN(nc, nf, 4, 2, 1)]
        tnf = nf
        for i in range(1, n_layer):
            model += [ConvLReluBN(tnf, tnf*2, 4, 2, 1)]  # every layer doubles features and halves img_size
            tnf *= 2
        self.net = nn.Sequential(*model)


class Upsampler(BaseModule):
    def __init__(self, n_layer=2, mode='nearest'):
        super(Upsampler, self).__init__()
        model = []
        tnf = nf * 2**(n_layer-1)
        for i in range(1, n_layer):
            model += [UpConvLReluBN(tnf, tnf/2, 3, 1, 1, mode)]
            tnf /= 2
        model += [nn.Upsample(scale_factor=2, mode=mode)]
        assert tnf == nf, '2**n_layer is not the upsampling factor'
        model += [nn.Conv2d(nf, nc, 3, 1, 1, bias=True)]  # no batchnorm afterwards -> use bias
        # model += [nn.Conv2d(nf, nc, 5, 1, 2, bias=True)]  # can also use this, does it look crispier? -> fix GAN loss
        model += [nn.Tanh()]
        self.net = nn.Sequential(*model)


class Deconver(BaseModule):
    def __init__(self, n_layer=2):
        super(Deconver, self).__init__()
        model = []
        tnf = nf * 2**(n_layer-1)
        for i in range(1, n_layer):
            model += [DeConvLReluBN(tnf, tnf/2, 4, 2, 1)]
            tnf /= 2
        assert tnf == nf, '2**n_layer is not the upsampling factor'
        model += [nn.ConvTranspose2d(nf, nc, 4, 2, 1, bias=True)]  # no batchnorm afterwards -> use bias
        model += [nn.Tanh()]
        self.net = nn.Sequential(*model)


##############################################################################################################
# Building Blocks
##############################################################################################################

class Discriminator(BaseModule):
    def __init__(self, n_layer=4):
        super(Discriminator, self).__init__()
        model = []
        model += [Reducer(n_layer)]
        model += [nn.Conv2d(nf * 2**(n_layer-1), 1, img_size / 2**n_layer, 1, 0)]  # effectively dense layer
        model += [nn.Sigmoid()]
        self.net = nn.Sequential(*model)


class Generator(BaseModule):
    def __init__(self, n_layer=4):
        super(Generator, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(nz, nf * 2**(n_layer-1), img_size / 2**n_layer, 1, 0)]
        # model += [Upsampler(n_layer, 'nearest')]
        model += [Deconver(n_layer)]
        self.net = nn.Sequential(*model)


class Encoder(BaseModule):  # like Discriminator but last layer to latent dimension instead of binary
    def __init__(self, n_layer=4):
        super(Encoder, self).__init__()
        self.net = Reducer(n_layer)
        self.mu = nn.Conv2d(nf * 2**(n_layer-1), nz, img_size / 2**n_layer, 1, 0)  # effectively linear layer
        # self.logvar = nn.Conv2d(nf * 2**(n_layer-1), nz, img_size / 2**n_layer, 1, 0)  # linear layer

    def forward(self, x):
        x = self.net(x)
        mu = F.tanh(self.mu(x))  # todo careful here: tanh restricts latent space to hypersphere!
        # logvar = F.tanh(self.logvar(x))
        return mu #, logvar


##############################################################################################################
# Assemble parts in complete architectures
##############################################################################################################

class VAE(nn.Module):
    def __init__(self, n_layer):
        super(VAE, self).__init__()
        self.encoder = Encoder(n_layer).cuda()  # todo do I need the cuda here?
        self.generator = Generator(n_layer).cuda()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparametrize(mu, logvar)
        x = self.generator(z)
        return x, mu, logvar


class BigBottle(nn.Module):
    def __init__(self, n_layer=4):
        super(BigBottle, self).__init__()
        self.encoder = Reducer(n_layer).cuda()
        self.generator = Deconver(n_layer).cuda()

    def forward(self, a, b):
        a_mini = self.encoder(a)
        b_mini = self.encoder(b)
        x_mini = (a_mini + b_mini) / 2
        x = self.generator(x_mini)
        return x

    def reconstruct(self, x):
        x = self.encoder(x)
        x = self.generator(x)
        return x


class TripletUNet(nn.Module):
    def __init__(self, n_layer=4):
        super(TripletUNet, self).__init__()
        self.background = Reducer(n_layer, nf/2)
        # to U-Net
        self.down0 = ConvLReluBN(nc, nf/2, 4, 2, 1)
        self.down1 = ConvLReluBN(nf/2, nf, 4, 2, 1)
        self.down2 = ConvLReluBN(nf, nf * 2, 4, 2, 1)
        self.down3 = ConvLReluBN(nf * 2, nf * 4, 4, 2, 1)  # nf * 4 = 128

        self.up0 = DeConvLReluBN(nf * 8, nf * 2, 4, 2, 1)
        self.up1 = DeConvLReluBN(nf * 4, nf, 4, 2, 1)
        self.up2 = DeConvLReluBN(nf * 2, nf, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(nf, nc, 4, 2, 1, bias=True)  # no batchnorm afterwards -> use bias

        self.pose = Encoder(n_layer)
        # nf_red = nf * 2 ** (n_layer - 1)

        self.mu = nn.Conv2d(nf * 2 ** (n_layer - 1), nz, img_size / 2 ** n_layer, 1, 0)  # effectively linear layer
        self.logvar = nn.Conv2d(nf * 2 ** (n_layer - 1), nz, img_size / 2 ** n_layer, 1, 0)  # linear layer
        self.pose_estimator = nn.ConvTranspose2d(nz, nf * 2 ** (n_layer - 2), img_size / 2 ** n_layer, 1, 0)  # use nf/2
        self.generator = Deconver(n_layer)

    def forward(self, a, b, sample=True):
        # pose estimation
        z_a, _ = self.pose(a)  # don't mind about log_var
        z_b, _ = self.pose(b)  # weight sharing
        # bg, me
        mu = (z_b - z_a) / 2  # from before: makes sure that no background is in pose
        # mu = (z_a + z_b) / 2
        # logvar = (logvar_a + logvar_b) / 2  # todo use logvar
        # if sample:
        #     mu = reparametrize(mu, logvar)
        pose = self.pose_estimator(mu)

        # U-net for background
        d0 = self.down0(a)  # a = 128, 3, 128, 128  , d0: 16
        d1 = self.down1(d0)  # 32
        d2 = self.down2(d1)  # 64
        back = self.down3(d2)  # 128

        assert(back.data.shape == pose.data.shape)
        x = torch.cat((back, pose), 1)  # 256
        u2 = self.up0(x)  # 64
        assert(u2.data.shape == d2.data.shape)
        x = torch.cat((u2, d2), 1)
        u1 = self.up1(x)
        assert(u1.data.shape == u1.data.shape)

        x = torch.cat((u1, d1), 1)
        u0 = self.up2(x)
        # x = torch.cat((u0, d0), 1)
        x = self.up3(u0)
        return x

    def reconstruct(self, x):
        return self.forward(x, x)


class TripletPro(nn.Module):
    def __init__(self, n_layer=4):
        super(TripletPro, self).__init__()
        self.appearance = Reducer(n_layer, nf / 2)
        self.pose = Encoder(n_layer)

        nf_red = nf * 2 ** (n_layer - 1)
        self.pose_estimator = nn.ConvTranspose2d(nz, nf_red / 2, img_size / 2 ** n_layer, 1, 0)  # use nf/2

        self.generator = Deconver(n_layer)

    def generate(self, pose, appear):
        pose = self.pose_estimator(pose)
        x = torch.cat((appear, pose), 1)
        x = self.generator(x)
        return x

    def forward(self, frame0, frame2, frame_rand, sample=False):
        p_0 = self.pose(frame0)  # don't mind about log_var
        p_2 = self.pose(frame2)  # weight sharing
        p_1 = (p_0 + p_2) / 2  # pose space should be locally linear (as any manifold by definition)

        a_rand = self.appearance(frame_rand)

        # if sample:  # reparametrize  # todo resolve NaN loss error
        #     std = torch.sqrt(torch.pow(p_0 - p_2, 2))  # element-wise norm as standard deviation
        #     eps = Variable(std.data.new(std.size()).normal_()).cuda()
        #     p_1 = eps.mul(std).add_(p_1)

        x = self.generate(p_1, a_rand)
        return x

    def reconstruct(self, x):
        return self.forward(x, x, x)


# how to build a z processing:
# self.pose_estimator = nn.ConvTranspose2d(nz, tnf / 2, img_size / 2 ** n_layer, 1, 0)  # use nf/2
# pose_a = self.pose_estimator(p_a)
# pose_b = self.pose_estimator(p_b)

# x_mini = torch.cat((a_rand, pose_a, pose_b), 1)
#
# tnf = nf * 2 ** (n_layer - 1)  # 128 for nf = 16
# self.generator = nn.Sequential(DeConvLReluBN(2 * tnf / 2, tnf / 2, 4, 2, 1),
#                                DeConvLReluBN(tnf / 2, tnf / 4, 4, 2, 1),
#                                DeConvLReluBN(tnf / 4, tnf / 8, 4, 2, 1),
#                                nn.ConvTranspose2d(tnf / 8, nc, 4, 2, 1, bias=True), nn.Tanh())
