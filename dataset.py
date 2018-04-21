import torch
import os
from skimage import io
from torchvision import transforms, datasets, utils
import config
import random

which_dataset = config.which_dataset  # 'mnist' for mnist
batch_size = config.batch_size
img_size = config.img_size  # width and height of image in pixels
nc = config.nc  # number of color channels
n_workers = config.num_workers
use_triplet = config.use_triplet


class VideoSequenceData(torch.utils.data.Dataset):
    """
    Dataset of image sequences
    :param root_dir: path to directory with sequences as subfolders and images within those subfolders
    :param transform: apply image transformation
    :return Dataset
    """

    def __init__(self, root_dir, transform=None, subset=False, which_frames=(0, 1, 2)):
        self.root_dir = root_dir
        self.transform = transform
        self.subset = subset  # bool: is this dataset for triplets or images?
        self.sequence_paths = []
        self.image_paths = []
        self.sequence_lengths = []
        self.sequences = []
        for (path, folders, _) in os.walk(root_dir):
            self.sequence_paths.extend(os.path.join(path, name) for name in folders)
        for i, sequence in enumerate(self.sequence_paths):
            for (path, _, files) in os.walk(sequence):
                    videos = [os.path.join(path, name) for name in files]
                    self.image_paths.extend(videos)
                    self.sequences.append(videos)
            self.sequence_lengths.append(len(files))

        subset_length = max(which_frames)
        self.num_subsets = sum(self.sequence_lengths) - subset_length * len(self.sequences)  # start and end not usable as triplet
        self.subsets = []
        for seq in self.sequences:

            for i in range(len(seq)-subset_length):  # for every subset
                sub = []
                for frame_idx in which_frames:  # extract all indicated frames
                    sub.append(seq[frame_idx+i])
                random_frame = random.choice(seq)  # choose random frame from sequence
                sub.append(random_frame)
                self.subsets.append(sub)

    def __len__(self):
        if self.subset:
            length = self.num_subsets
        else:
            length = len(self.image_paths)
        return length

    def __getitem__(self, idx):
        if self.subset:
            item = self.get_subset(idx)
        else:
            item = self.get_image(idx)
        return item

    def get_image(self, idx, img_path=None):
        """Can either use index (idx) or the direct path (img_path) to return image"""
        if idx is not None:
            img_path = self.image_paths[idx]
        else:
            assert(img_path is not None)
        image = io.imread(img_path)
        if self.transform:
            image = self.transform(image)
        return image

    def get_subset(self, idx):
        """:return list of three following images in a video sequence"""
        assert (idx in range(self.num_subsets))
        image_idx = self.subsets[idx]
        frames = []
        for i in image_idx:
            frames.append(self.get_image(idx=None, img_path=i))

        return frames


assert which_dataset in ['olympic', 'kth']
# if which_dataset == 'mnist':
#     assert(img_size == 28)
#     assert(nc == 1)
#     transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
#     trainset = datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
#     testset = datasets.MNIST(root='./data/', train=False, download=True, transform=transform)
#

# Create transform
t_list = [transforms.ToPILImage()]
if nc == 1:  # use Grayscale
    t_list.append(transforms.Grayscale(1))
else:
    pass  # todo color augmentation
t_list.append(transforms.Resize([img_size, img_size]))
if not use_triplet:  # no flipping in triplets
    t_list.append(transforms.RandomHorizontalFlip())
t_list.append(transforms.ToTensor())
transform = transforms.Compose(t_list)

# Choose dataset
dset_path = '../../dsets/'
if which_dataset == 'kth':
    test_dir = '{}KTH/running'.format(dset_path)
    train_dir = '{}KTH/running'.format(dset_path)

elif which_dataset == 'olympic':
    test_dir = '{}OlympicSports/long_jump_test'.format(dset_path)
    train_dir = '{}OlympicSports/long_jump'.format(dset_path)


# Define datasets
if use_triplet:
    trainset = VideoSequenceData(train_dir, transform, subset=True, which_frames=(0, 2, 4))
    testset = VideoSequenceData(test_dir, transform, subset=True, which_frames=(0, 2, 4))
else:
    trainset = VideoSequenceData(train_dir, transform, subset=False)
    testset = VideoSequenceData(test_dir, transform, subset=False)

# Choose data loaders
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=n_workers)


def load_batch(number):
    loader = torch.utils.data.DataLoader(trainset, batch_size=number, shuffle=True, num_workers=n_workers)

    if which_dataset == 'mnist':
        batch, _ = next(iter(loader))  # for GAN do not want classes
    else:
        batch = next(iter(loader))
    return batch

