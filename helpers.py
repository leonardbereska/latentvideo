import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torchvision
from torch.autograd import Variable
import torch
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from shutil import copy2
import os

import dataset
import models


# todo seems to be running on cpu?!
def add_gaussian_noise(images, std):  # for instance noise
    img_shape = images.shape
    instance_noise = Variable(torch.normal(means=torch.zeros(img_shape),
                                           std=torch.ones(img_shape) * std)).cuda()
    return images + instance_noise


def apply_tsne_img(model):
    """
    Manifold approximation using tSNE
    features: feature vectors to be presented in two dimensions
    """
    nc = dataset.nc
    img_size = dataset.img_size  # is this a good idea?

    images = dataset.load_batch(1000)
    images = images.view(-1, nc, img_size, img_size)
    images = images.clamp(0, 1)
    numpy_images = images.numpy()
    numpy_images = np.transpose(numpy_images, (0, 2, 3, 1))
    images = Variable(images)
    features, _ = model.extract_features(images)  # only take mean as feature
    features = features.data.numpy()
    images = numpy_images

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    vis_xy = tsne.fit_transform(features)

    x_min, x_max = np.min(vis_xy, 0), np.max(vis_xy, 0)
    vis_xy = (vis_xy - x_min) / (x_max - x_min)  # scale to interval (0, 1)
    vis_x = vis_xy[:, 0]
    vis_y = vis_xy[:, 1]
    print("Plotting t-SNE embedding")


    # fig, ax = plt.subplots()
    # plt.plot(vis_x, vis_y, '.')

    max_width = max([image.shape[0] for image in images])
    max_height = max([image.shape[1] for image in images])

    # get max, min coords
    x_min, x_max = vis_x.min(), vis_x.max()
    y_min, y_max = vis_y.min(), vis_y.max()

    # Fix the ratios
    res = 4000
    sx = (x_max - x_min)
    sy = (y_max - y_min)
    if sx > sy:
        res_x = int(sx / float(sy) * res)
        res_y = res
    else:
        res_x = res
        res_y = int(sy / float(sx) * res)

    # impaint images
    canvas = np.ones((res_x + max_width, res_y + max_height, 3))
    x_coords = np.linspace(x_min, x_max, res_x)
    y_coords = np.linspace(y_min, y_max, res_y)
    for x, y, image in zip(vis_x, vis_y, images):
        w, h = image.shape[:2]
        x_idx = np.argmin((x - x_coords) ** 2)
        y_idx = np.argmin((y - y_coords) ** 2)
        try:
            canvas[x_idx:x_idx + w, y_idx:y_idx + h] = image
        except:
            print('Image out of borders.... skip!')

    # plot image
    # fig = plt.figure()
    plt.imshow(canvas)
    plt.xticks([]), plt.yticks([])
    plt.title("t-SNE of Video Frames")
    plt.savefig('{}-tsne.png'.format('sports'))
    plt.show(block=True)


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)  # TODO should this be not rather (-1, 1)
    return inp


def plot_reconstruct(encoder, generator):
    """
    Visualizes how the Autoencoder transforms the dataset images
    :return: plot of dataset batch and corresponding autoencoded images
    """

    encoder.eval()
    generator.eval()
    # Get a batch of test data
    data = dataset.load_batch(4)
    data = Variable(data, volatile=True).cuda()

    # # Generate images
    # shape = (4, 1, 256)
    # mu = torch.FloatTensor(np.random.normal(size=shape))
    # var = torch.FloatTensor(np.zeros(shape))
    # mu, var = Variable(mu), Variable(var)

    # Make grid of numpy images
    # model.eval()
    input = data.data.cpu()
    # output_tensor, _, _ = model(data)  # mu and logvar not needed
    mu, logvar = encoder(data)
    z = models.reparametrize(mu, logvar)
    output_tensor = generator(z)
    output = output_tensor.data.cpu()
    in_grid = convert_image_np(torchvision.utils.make_grid(input, 2))
    out_grid = convert_image_np(torchvision.utils.make_grid(output, 2))

    # Plot the results side-by-side
    fig = plt.gcf()
    plt.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(in_grid)
    ax1.set_title('Dataset Images')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(out_grid)
    ax2.set_title('Reconstructed Images')
    plt.suptitle('VAE Visualization')
    plt.pause(0.0001)  # pause to update plot
    plt.show(block=False)  # block : don't pause code after plotting


# TODO what happens for img_size = 128, why does the loss explode? logvar?
def vae_loss(recon_x, x, mu, logvar, batch_size, img_size, nc):
    """
    VAE Loss
    :param recon_x: image reconstructions
    :param x: images
    :param mu, logvar: outputs of your encoder
    :param batch_size:
    :param img_size: width, respectively height of you images
    :param nc: number of image channels
    :return: loss
    """
    L1 = F.l1_loss(recon_x, x)  # TODO .cuda() here?
    # MSE = F.mse_loss(recon_x, x)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Normalize
    KLD /= batch_size * img_size * img_size * nc

    return L1 + KLD  # MSE + KLD


def reconstruct(model, triplet, frame_number, img_size):
    """Reconstructs or interpolates frame
    if frame 0 or 2 reconstruct
    if frame 1: interpolate"""
    assert (frame_number in [0, 1, 2])
    if frame_number in [0, 2]:
        frame = Variable(triplet[frame_number].view(1, 3, img_size, img_size)).cuda()
        mu, log_var = model.extract_features(frame)
    else:
        assert (frame_number == 1)
        frame1 = Variable(triplet[0].view(1, 3, img_size, img_size)).cuda()
        frame2 = Variable(triplet[2].view(1, 3, img_size, img_size)).cuda()
        mu1, log_var1 = model.extract_features(frame1)
        mu2, log_var2 = model.extract_features(frame2)
        mu = mu1 + mu2 / 2
        # log_var = torch.sqrt(log_var1**2 + log_var2**2)  # construct covariance matrix
        log_var = log_var1 * 0  # generate
    model.eval()
    reconst_img = model.generate(mu, log_var).cpu()
    pic = convert_image_np(reconst_img.data.view(3, img_size, img_size))
    return pic


# def plot_triplet():
#     triplet = dataset.trainset.get_triplet()
#     to_image = transforms.ToPILImage()
#     images = [to_image(t) for t in triplet]
#     plt.subplot(231)
#     plt.xticks([]), plt.yticks([])
#     plt.imshow(images[0])
#     plt.title('First')
#     plt.subplot(232)
#     plt.xticks([]), plt.yticks([])
#     plt.imshow(images[1])
#     plt.title('Middle')
#     plt.subplot(233)
#     plt.xticks([]), plt.yticks([])
#     plt.imshow(images[2])
#     plt.title('Last')
#     plt.subplot(234)
#     plt.xticks([]), plt.yticks([])
#     plt.imshow(reconstruct(triplet, 0))
#     plt.title('Reconstruction')
#     plt.subplot(235)
#     plt.xticks([]), plt.yticks([])
#     plt.imshow(reconstruct(triplet, 1))
#     plt.title('Interpolation')
#     plt.subplot(236)
#     plt.xticks([]), plt.yticks([])
#     plt.imshow(reconstruct(triplet, 2))
#     plt.title('Reconstruction')
#     plt.show(block=True)


def init(gpu, file_name, experiment_name):
    """
    Initialize experiment and save meta data
    :param gpu: number of gpu device
    :param file_name: file to be executed
    :param experiment_name: name of the experiment
    :return: writer for tensorboard, directory in which all relevant files are saved
    """
    # set gpu
    if torch.cuda.is_available():
        print('Using CUDA on GPU {}'.format(gpu))
        torch.cuda.set_device(gpu)
    else:
        print('CUDA not available')

    # init file names
    folder = 'experiments'
    save_dir = '{}/{}/{}'.format(folder, file_name, experiment_name)  # all relevant data for experiment stored here
    writer = SummaryWriter('{}'.format(save_dir))  # tensorboard writer

    # write config data to tensorboard
    with open('config.py', 'r') as f:
        for line in f:
            print(line)
            writer.add_text(line, line, 0)

    # save all files to experiment dir
    mypath = '../Master'
    src_files = os.listdir(mypath)
    for file_name in src_files:
        full_file_name = os.path.join(mypath, file_name)
        if os.path.isfile(full_file_name):
            copy2(full_file_name, save_dir)

    return writer, save_dir
