import os
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from .base import BaseModel

# define pytorch device - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model parameters
NUM_EPOCHS = 90  # original paper
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227  # pixels
NUM_CLASSES = 1000  # 1000 classes for imagenet 2012 dataset
DEVICE_IDS = [0, 1, 2, 3]  # GPUs to use
# modify this to point to your data directory
INPUT_ROOT_DIR = 'alexnet_data_in'
TRAIN_IMG_DIR = 'alexnet_data_in/imagenet'
OUTPUT_DIR = 'alexnet_data_out'
LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints

# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class AlexNet(BaseModel):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self, input_mean, input_std, output_mean, output_std,
                 num_of_input, num_of_output, dropout):
        """
        Define and allocate layers for this neural net.

        Args:
            num_classes (int): number of classes to predict with this model
        """
        super(AlexNet, self).__init__("Alexnet", input_mean, input_std, output_mean,
                                  output_std, num_of_input, num_of_output)
        # input size should be : (b x 3 x 240 x 180)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=num_of_input,
                      out_channels=96,
                      kernel_size=11,
                      stride=4),  # (b x 96 x 60 x 45)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75,
                                 k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 30 x 22)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 30 x 22)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 15 x 11)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 15 x 11)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 15 x 11)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 15 x 11)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 4)
        )
        self.net_output_size = 256 * 6 * 4
        # regressor is just a name for linear layers
        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=(self.net_output_size), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_of_output),
        )
        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        """
        Pass the input through the net.

        Args:
            x (Tensor): input tensor

        Returns:
            output (Tensor): output tensor
        """
        x = self.normalize_input(x)
        x = self.net(x)
        # print(f"x shape {x.shape}")
        x = x.view(-1, self.net_output_size
                   )  # reduce the dimensions for linear layer input
        x  = self.regressor(x)
        x = self.unnormalize_output(x)
        x = self.clamp(x)
        return x


def make_alexnet(input_mean, input_std, output_mean, output_std, num_of_input,
                 num_of_output, dropout):

    return AlexNet(input_mean, input_std, output_mean, output_std,
                   num_of_input, num_of_output, dropout)
