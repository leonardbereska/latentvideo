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
# assert(file_name == 'triplet' or file_name == 'bigbottle' or file_name == 'tripletpro')
writer, save_dir = helpers.init(gpu, file_name, experiment_name)

# model = models.Triplet().cuda()
# model = models.BigBottle2(n_layer=n_layer).cuda()
model = models.TripletPro().cuda()
model.apply(models.init_weights)  # xavier init
optimizer = optim.Adam(model.parameters(), lr=lr)


def run(n_epochs):

    for epoch in range(n_epochs):

        for n, (frame0, frame1, frame2, frame_rand) in enumerate(dataset.train_loader):
            niter = epoch * len(dataset.train_loader) + n  # count gradient updates
            model.train()
            frame0 = Variable(frame0).cuda()
            frame1 = Variable(frame1).cuda()
            frame2 = Variable(frame2).cuda()
            frame_rand = Variable(frame_rand).cuda()

            # Appearance constancy loss
            # a1 = model.appearance(frame0)
            # a2 = model.appearance(frame_rand)
            # loss_appear = F.l1_loss(a1, a2)  # two frames in video should have same appearance

            # Pose constancy loss
            # p1 = model.pose(frame0)
            # frame_trans = frame0  # insert some transform here e.g. contrast, color inversion, small transl/rotations
            # p2 = model.pose(frame_trans)
            # loss_pose = F.l1_loss(p1, p2)  # pose should not change under transformations

            # Reconstruction Loss
            optimizer.zero_grad()
            # output, mu, log_var = model(frame0, frame2)
            # loss = helpers.vae_loss(output, frame1, mu=mu, logvar=log_var, batch_size=batch_size, img_size=img_size,
            #                         nc=nc)
            output = model(frame0, frame2, frame_rand)
            loss_reconst = F.l1_loss(output, frame1)  # TODO make a proper VAE loss

            loss = loss_reconst  #  + loss_appear  # + loss_pose

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
            for n, (frame0, frame1, frame2, frame_rand) in enumerate(dataset.test_loader):
                frame0 = Variable(frame0).cuda()
                frame1 = Variable(frame1).cuda()
                frame2 = Variable(frame2).cuda()
                frame_rand = Variable(frame_rand).cuda()
                # output, mu, log_var = model(frame0, frame2)
                # loss = helpers.vae_loss(output, frame1, mu=mu, logvar=log_var, batch_size=batch_size, img_size=img_size,
                #                         nc=nc)
                output = model(frame0, frame2, frame_rand)
                loss = F.l1_loss(output, frame1)
                test_loss += loss.data[0]
            test_loss /= len(dataset.test_loader)
            writer.add_scalar('Loss/Test', test_loss, epoch)

            # test reconstruction quality for images from train and test set

            # TODo new eval.py for inspecting latent space
            # phases = ['train', 'test']
            phases = ['train']
            for phase in phases:
                if phase == 'train':
                    evalset = dataset.trainset
                else:
                    evalset = dataset.testset

                # Test triplet reconstruction
                # get random subset
                idx = np.random.choice(range(evalset.num_subsets))  # random index of triplet
                frames = evalset.get_subset(idx)  # triplet from train data

                inputs = list(f.view([1] + [i for i in f.shape]) for f in frames)  # format for batch
                frames = list(Variable(frame).cuda() for frame in inputs)

                outputs = list()
                outputs.append(model.reconstruct(frames[0]))
                outputs.append(model.forward(frames[0], frames[2], frames[3]))
                outputs.append(model.reconstruct(frames[2]))
                outputs.append(model.reconstruct(frames[3]))  # also reconstruct random frame
                outputs = [out.data.cpu() for out in outputs]
                show_images(inputs+outputs, 4, 'Interpolation', epoch)

                # Test pose and appearance switch
                # video = np.random.choice(evalset.sequences)  # same video
                # a, b = np.random.choice(video, 2)
                # a = evalset.get_image(None, img_path=a)
                # b = evalset.get_image(None, img_path=b)
                video1, video2 = np.random.choice(evalset.sequences, 2)  # different video
                a = np.random.choice(video1)
                b = np.random.choice(video2)
                a = evalset.get_image(None, img_path=a)
                b = evalset.get_image(None, img_path=b)

                a = a.view([1] + [i for i in a.shape])
                b = b.view([1] + [i for i in b.shape])
                p_a = model.pose(Variable(a).cuda())
                p_b = model.pose(Variable(b).cuda())
                a_a = model.appearance(Variable(a).cuda())
                a_b = model.appearance(Variable(b).cuda())
                x_ab = model.generate(p_a, a_b)  # pose a, appearance b
                x_ba = model.generate(p_b, a_a)
                x_ab = x_ab.data.cpu()
                x_ba = x_ba.data.cpu()
                show_images([a, b, x_ab, x_ba], 2, 'Switch Pose/Appearance', epoch)

                # Test interpolation
                length = 5
                seq = video1[0:length]
                seq = [evalset.get_image(None, img_path=path) for path in seq]
                seq = [img.view([1] + [i for i in img.shape]) for img in seq]
                appear = model.appearance(Variable(seq[0]).cuda())
                p_init = model.pose(Variable(seq[0]).cuda())
                p_end = model.pose(Variable(seq[-1]).cuda())
                alpha = [float(i) / (length-1) for i in range(0, length)]
                poses = [alpha[i] * p_init + (1-alpha[i]) * p_end for i in range(0, length)]
                images = [model.generate(p, appear) for p in poses]
                show_images(images, length, 'Linear Interpolation in Pose space', epoch)



def show_images(img_list, how_many_in_one_row, description, iter):
    img_list = torch.cat(img_list, 0)
    grid = helpers.convert_image_np(torchvision.utils.make_grid(img_list, how_many_in_one_row))
    writer.add_image(description, grid, iter)



        # torch.save(model.state_dict(), '{}/triplet.pkl'.format(save_dir))


run(num_epochs)
