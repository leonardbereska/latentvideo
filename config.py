# Triplet testing U-Net
print_output = False
file_name = 'tripletpro'
gpu = 1
img_size = 64
nf = 16  # number of features
nz = 200  # dimension of latent space
n_layer = 4  # number of generator/encoder layers
batch_size = 64
nc = 3  # number of channels
log_interval = 2
num_epochs = 100
lr = 0.001  # 10^-3 is fast, 10^-4 is save
which_dataset = 'triplet'
num_workers = 16
# experiment_name = 'img{}_nf{}_layers{}'.format(img_size, nf, n_layer)
# experiment_name = 'img{}_nf{}_nz{}'.format(img_size, nf, nz)
experiment_name = 'test0'