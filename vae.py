import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import numpy as np

import dataset
import helpers
import models
from config import *
assert(file_name == 'vae')
writer, save_dir = helpers.init(gpu, file_name, experiment_name)


def run(n_epochs):
    is_best = np.inf

    for epoch in range(n_epochs):

        for n, images in enumerate(dataset.train_loader):
            niter = epoch * len(dataset.train_loader) + n  # count gradient updates

            # training
            model.train()
            images = Variable(images).cuda()
            optimizer.zero_grad()
            output, mu, log_var = model(images)
            loss = helpers.vae_loss(output, images, mu=mu, logvar=log_var, batch_size=batch_size, img_size=img_size, nc=nc)
            loss.backward()
            optimizer.step()
            train_loss = loss.data[0]
            writer.add_scalar('Loss/Train', train_loss, niter)

        if epoch % log_interval == 0:

            # testing
            model.eval()
            test_loss = 0
            for n, images in enumerate(dataset.test_loader):
                images = Variable(images).cuda()
                output, mu, log_var = model(images)
                test_loss += helpers.vae_loss(output, images, mu, log_var, batch_size, img_size, nc).data[0]
            test_loss /= len(dataset.test_loader)   # average over all iterations
            writer.add_scalar('Loss/Test', test_loss, epoch)

            if test_loss < is_best:
                is_best = test_loss
                torch.save(model.state_dict(), '{}/vae.pkl'.format(save_dir))
                writer.add_text('best epoch', 'saved model at epoch {}'.format(epoch), 0)

            if print_output:
                print("Epoch [{}/{}], Gradient Step: {}, Train Loss: {:.4f}, Test Loss: {:.4f}"
                    .format(epoch, num_epochs, (epoch + 1) * len(dataset.train_loader), train_loss, test_loss))

            # inspect reconstruction quality
            n_samples = 4
            data = dataset.load_batch(n_samples)  # Get a batch of test data
            input = Variable(data, volatile=True).cuda()  # TODO what does volatile do here?
            output, _, _ = model(input)
            noise = Variable(torch.randn(n_samples, nz, 1, 1).cuda())  # fake_images = generator(noise)
            generations = model.generator(noise)

            sequence = torch.cat((input.data.cpu(), output.data.cpu(), generations.data.cpu()), 0)
            grid = helpers.convert_image_np(torchvision.utils.make_grid(sequence, n_samples))

            writer.add_image('Input, Reconstruction, Generation', grid, epoch)


model = models.VAE(n_layer).cuda()
model.apply(models.init_weights)  # xavier init

optimizer = optim.Adam(model.parameters(), lr=lr)

# model.load_state_dict(torch.load('{}/vae.pkl'.format(save_dir)))

run(num_epochs)


