import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np

import dataset
import helpers
import models
from config import *
assert(file_name == 'vae-gan')
writer, save_dir = helpers.init(gpu, file_name, experiment_name)


use_ganloss = True  # TODO can I use that here?
use_vaeloss = True
use_instancenoise = False
iter_decay = 1000.  # iterations after which instance noise is at 1/e


# models  # TODO to one model?
netD = models.Discriminator().cuda()
netG = models.Generator().cuda()
netE = models.Encoder().cuda()

# weight initialization
netD.apply(models.init_weights)  # xavier init
netE.apply(models.init_weights)  # xavier init
netG.apply(models.init_weights)  # xavier init

criterion = nn.BCELoss().cuda()
optD = torch.optim.Adam(netD.parameters(), lr=lr)
optG = torch.optim.Adam(netG.parameters(), lr=lr)
optE = torch.optim.Adam(netE.parameters(), lr=lr)


def sample(n_samples):
    noise = Variable(torch.randn(n_samples, nz, 1, 1)).cuda()  # fake_images = generator(noise)
    fake_images = netG(noise)
    return fake_images


def reconstruct(images):
    mu, logvar = netE(images)
    z = models.reparametrize(mu, logvar)
    fake_images = netG(z)
    return fake_images, mu, logvar


def train_gan(images, noise_level):  # original GAN loss
    real_labels = Variable(torch.ones(images.size(0))).cuda().view(-1, 1, 1, 1)
    fake_images, _, _ = reconstruct(images)

    # Update D: maximize log(D(x)) + log(1 - D(G(z)))
    netD.zero_grad()

    # add instance noise
    noisy_images = images
    if use_instancenoise:
        noisy_images = helpers.add_gaussian_noise(images, noise_level)
        fake_images = helpers.add_gaussian_noise(fake_images, noise_level)

    # train with real images
    real_outputs = netD(noisy_images)
    errD_real = criterion(real_outputs, real_labels)
    errD_real.backward()
    # train with fake images
    fake_labels = Variable(torch.zeros(images.size(0))).cuda().view(-1, 1, 1, 1)
    fake_outputs = netD(fake_images.detach())  # todo is detach correct here?
    errD_fake = criterion(fake_outputs, fake_labels)
    errD_fake.backward()

    errD = errD_real + errD_fake
    optD.step()

    # Update G: maximize log(D(G(z)))
    optG.zero_grad()
    fake_images, _, _ = reconstruct(images)  # use fake images from above (.detach()
    outputs = netD(fake_images)
    errG = criterion(outputs, real_labels)
    errG.backward()
    optG.step()

    return errD.data[0], errG.data[0], real_outputs.data.mean(), fake_outputs.data.mean()


def train_gan_disfeature(images, d_layer):
    fake_images, _, _ = reconstruct(images)

    extractor = models.FeatureExtractor(netD, d_layer)
 # Update D: maximize log(D(x)) + log(1 - D(G(z)))
    netD.zero_grad()

    # train with real images
    real_outputs = netD(images)
    errD_real = criterion(real_outputs, real_labels)
    errD_real.backward()
    # train with fake images
    fake_labels = Variable(torch.zeros(images.size(0))).cuda().view(-1, 1, 1, 1)
    fake_outputs = netD(fake_images.detach())  # todo is detach correct here?
    errD_fake = criterion(fake_outputs, fake_labels)
    errD_fake.backward()

    errD = errD_real + errD_fake
    optD.step()

    # Update G: maximize log(D(G(z)))
    optG.zero_grad()
    fake_images, _, _ = reconstruct(images)  # use fake images from above (.detach()
    outputs = netD(fake_images)
    errG = criterion(outputs, real_labels)
    errG.backward()
    optG.step()

    return errD.data[0], errG.data[0], real_outputs.data.mean(), fake_outputs.data.mean()

def train_vae(images):
    # Update G and E: VAE
    optG.zero_grad()
    optE.zero_grad()
    fake_images, mu, logvar = reconstruct(images)  # todo can I save a forward pass here?
    vae_loss = helpers.vae_loss(fake_images, images, mu, logvar, batch_size, img_size, nc)
    vae_loss.backward()
    optG.step()
    optE.step()  # todo does this optimization work?: apparently
    return vae_loss.data[0]


# set number of epochs
num_batches = len(dataset.train_loader)

# encoder.load_state_dict(torch.load('{}/vaegan-encoder.pkl'.format(save_file)))
# generator.load_state_dict(torch.load('{}/vaegan-generator.pkl'.format(save_file)))
# discriminator.load_state_dict(torch.load('{}/vaegan-discriminator.pkl'.format(save_file)))

for epoch in range(num_epochs):
    for n, images in enumerate(dataset.train_loader):

        images = Variable(images).cuda()
        niter = epoch * len(dataset.train_loader) + n  # count gradient updates
        noise_level = np.exp(-niter / iter_decay)  # instance noise level
        netD.train()
        netG.train()
        netE.train()

        vae_loss = train_vae(images)
        d_loss, g_loss, real_score, fake_score = train_gan_disfeature(images, 2)

        # write monitoring data to tensorboard
        writer.add_scalar('Loss/D', d_loss, niter)
        writer.add_scalar('Loss/G', g_loss, niter)
        writer.add_scalar('Loss/VAE', vae_loss, niter)
        writer.add_scalar('Score/Real', real_score, niter)
        writer.add_scalar('Score/Fake', fake_score, niter)
        writer.add_scalar('Instance Noise', noise_level, niter)

    if epoch % log_interval == 0:
        if print_output:
            print("Epoch [{}/{}], Step [{}/{}], Loss_g: {:.4f}, Loss_d: {:.4f}, Real Score: {:.2f}, Fake Score: {:.2f}"
                  .format(epoch, num_epochs, n, num_batches, g_loss,
                          d_loss, real_score, fake_score))
            print("VAE loss: {}".format(vae_loss))

        n_samples = 4
        netE.eval()
        netG.eval()

        data = Variable(dataset.load_batch(n_samples)).cuda()
        reconstructions, _, _ = reconstruct(data)

        originals = data.data.cpu()
        inputs = inputs = helpers.add_gaussian_noise(data, noise_level).data.cpu()
        fake_input = helpers.add_gaussian_noise(reconstructions, noise_level).data.cpu()
        reconstructions = reconstructions.data.cpu()
        generations = sample(n_samples).view(-1, nc, img_size, img_size).data.cpu()

        sequence = torch.cat((originals, inputs, fake_input, reconstructions, generations), 0)
        grid = helpers.convert_image_np(torchvision.utils.make_grid(sequence, n_samples))
        writer.add_image('Original, Input, Fake Input, Reconstruction, Generation', grid, epoch)

        torch.save(netG.state_dict(), '{}/generator.pkl'.format(save_dir))
        torch.save(netD.state_dict(), '{}/discriminator.pkl'.format(save_dir))
        torch.save(netE.state_dict(), '{}/encoder.pkl'.format(save_dir))


