import torch.optim as optim
import torchvision
import torch
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F

import dataset
import helpers
import models
from config import *
assert(file_name == 'triplet' or file_name == 'bigbottle' or file_name == 'tripletpro')
writer, save_dir = helpers.init(gpu, file_name, experiment_name)

# model = models.Triplet().cuda()
# model = models.BigBottle2(n_layer=n_layer).cuda()
model = models.TripletPro().cuda()
model.apply(models.init_weights)  # xavier init
optimizer = optim.Adam(model.parameters(), lr=lr)


def run(n_epochs):

    for epoch in range(n_epochs):

        for n, (frame0, frame1, frame2, random_frame) in enumerate(dataset.train_loader):
            niter = epoch * len(dataset.train_loader) + n  # count gradient updates
            model.train()
            frame0 = Variable(frame0).cuda()
            frame1 = Variable(frame1).cuda()
            frame2 = Variable(frame2).cuda()
            random_frame = Variable(random_frame).cuda()


            optimizer.zero_grad()
            # output, mu, log_var = model(frame0, frame2)
            # loss = helpers.vae_loss(output, frame1, mu=mu, logvar=log_var, batch_size=batch_size, img_size=img_size,
            #                         nc=nc)

            output = model(frame0, frame2, random_frame)
            loss = F.l1_loss(output, frame1)  # TODO make a proper VAE loss



            loss.backward()
            optimizer.step()
            train_loss = loss.data[0]
            writer.add_scalar('Loss/Train', train_loss, niter)

        if epoch % log_interval == 0:
            if print_output:
                print("Epoch [{}/{}], Gradient Step: {}, Train Loss: {:.4f}"
                      .format(epoch, n_epochs, (epoch + 1) * len(dataset.train_loader), train_loss))

            # test loss
            model.eval()
            test_loss = 0
            for n, (frame0, frame1, frame2, random_frame) in enumerate(dataset.test_loader):
                frame0 = Variable(frame0).cuda()
                frame1 = Variable(frame1).cuda()
                frame2 = Variable(frame2).cuda()
                random_frame = Variable(random_frame).cuda()
                # output, mu, log_var = model(frame0, frame2)
                # loss = helpers.vae_loss(output, frame1, mu=mu, logvar=log_var, batch_size=batch_size, img_size=img_size,
                #                         nc=nc)
                output = model(frame0, frame2, random_frame)
                loss = F.l1_loss(output, frame1)
                test_loss += loss.data[0]
            test_loss /= len(dataset.test_loader)
            writer.add_scalar('Loss/Test', test_loss, epoch)

            # test reconstruction quality for images from train and test set

            # TODo new eval.py
            phases = ['train', 'test']
            for phase in phases:
                if phase == 'train':
                    idx = np.random.choice(range(dataset.trainset.num_subsets))  # random index of triplet
                    frame0, frame1, frame2, random_frame = dataset.trainset.get_subset(idx)  # triplet from train data
                else:
                    idx = np.random.choice(range(dataset.testset.num_subsets))  # random index of triplet
                    frame0, frame1, frame2, random_frame = dataset.testset.get_subset(idx)  # triplet from train data

                frames = (frame0, frame1, frame2, random_frame)
                inputs = list(frame.view(1, frame.shape[0], frame.shape[1], frame.shape[2]) for frame in frames)
                frames = list(Variable(frame).cuda() for frame in inputs)

                out1 = model.forward(frames[0], frames[2], frames[3])
                out0 = model.reconstruct(frames[0])
                out2 = model.reconstruct(frames[2])
                out3 = model.reconstruct(frames[3])  # also reconstruct random frame

                sequence = inputs + list(out.data.cpu() for out in (out0, out1, out2, out3))
                sequence = torch.cat(sequence, 0)
                grid = helpers.convert_image_np(torchvision.utils.make_grid(sequence, 4))  # 3 for triplet
                writer.add_image('Original, Reconstruction/Interpolation {}'.format(phase), grid, epoch)

        # torch.save(model.state_dict(), '{}/triplet.pkl'.format(save_dir))


run(num_epochs)
