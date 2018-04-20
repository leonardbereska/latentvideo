import numpy as np
import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable

import dataset
import models
import helpers
from config import *
assert(file_name == 'gan')
writer, save_dir = helpers.init(gpu, file_name, experiment_name)


discriminator = models.Discriminator().cuda()
generator = models.Generator().cuda()

criterion = nn.BCELoss().cuda()  # cuda here?
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)


def sample(n_samples):
    noise = Variable(torch.randn(n_samples, nz, 1, 1)).cuda()  # fake_images = generator(noise)
    fake_images = generator(noise)
    return fake_images


def train_discriminator(discriminator, images, real_labels, fake_images, fake_labels):
    discriminator.zero_grad()
    outputs = discriminator(images)
    real_loss = criterion(outputs, real_labels)
    real_score = outputs

    outputs = discriminator(fake_images)
    fake_loss = criterion(outputs, fake_labels)
    fake_score = outputs

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data[0], real_score.data.mean(), fake_score.data.mean()


def train_generator(generator, discriminator_outputs, real_labels):
    generator.zero_grad()
    g_loss = criterion(discriminator_outputs, real_labels)  # TODO feature matching (feature of real data vs gen)
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data[0]


# set number of epochs and initialize figure counter
num_batches = len(dataset.train_loader)

# generator.load_state_dict(torch.load('{}/generator.pkl'.format(save_dir)))
# discriminator.load_state_dict(torch.load('{}/discriminator.pkl'.format(save_dir)))


for epoch in range(num_epochs):

    for n, images in enumerate(dataset.train_loader):
        niter = epoch * len(dataset.train_loader) + n  # count gradient updates
        images = Variable(images.cuda())
        real_labels = Variable(torch.ones(images.size(0)).cuda()).view(-1, 1, 1, 1)  # for ensuring same size in BCELoss

        # Sample from generator
        fake_images = sample(images.size(0))
        fake_labels = Variable(torch.zeros(images.size(0)).cuda()).view(-1, 1, 1, 1)

        # Train the discriminator
        d_loss, real_score, fake_score = train_discriminator(discriminator, images, real_labels, fake_images,
                                                             fake_labels)

        # Sample again from the generator and get output from discriminator
        fake_images = sample(images.size(0))
        outputs = discriminator(fake_images)

        # Train the generator
        g_loss = train_generator(generator, outputs, real_labels)  # use real_labels as fake_labels

        writer.add_scalar('Loss/G', g_loss, niter)
        writer.add_scalar('Loss/D', d_loss, niter)
        writer.add_scalar('Score/Fake', fake_score, niter)
        writer.add_scalar('Score/Real', real_score, niter)

    if epoch % log_interval == 0:
        if print_output:
            print("Epoch [{}/{}], Step [{}/{}], Loss_g: {:.4f}, Loss_d: {:.4f}, Real Score: {:.2f}, Fake Score: {:.2f}"
                .format(epoch, num_epochs, n, num_batches, g_loss,
                        d_loss, real_score, fake_score))
        # show generated images
        num_test_samples = 16
        test_images = sample(num_test_samples)
        test_images = torchvision.utils.make_grid(test_images.cpu().data, 4).numpy().transpose((1, 2, 0))
        test_images = np.clip(test_images, 0, 1)
        writer.add_image('Generations', test_images, epoch)

        torch.save(generator.state_dict(), '{}/generator.pkl'.format(save_dir))
        torch.save(discriminator.state_dict(), '{}/discriminator.pkl'.format(save_dir))


# Stabilize GAN Training Tricks
# from https://github.com/soumith/ganhacks

# 1. normalize inputs
# 2. modify Loss : fake -> real for Generator
# 3. use spherical z
# 4. batchnorm on only-real/only-fake minibatches
# 5. Avoid Sparse Gradients: LeakyReLU, Conv/ConvTranspose ## LeakyReLU with 0.2 did make it work
# 6. Soft + Noisy Labels  # TODO
# 7. DCGAN or Hybrid Loss # TODO try vae loss
# 8. Experience Replay, other RL tricks
# 9. Use Adam of G (SGD for D)  # TODO SGD for D
# 10. Fail fast: D loss to 0, large gradients # TODO check norm of gradient
# 11. Dont balance via schedule # maybe with loss threshold
# 12. use labels: auxiliary GAN
# 13. instance noise, add noise to inputs of D # TODO
# 17. use dropout (0.5)

# no batchnorm in discriminator