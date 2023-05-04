
import os
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import copy
import logging
import random
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import time
from glob import glob
from torch import Tensor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torchsummary import summary

# 30 SRM filtes

# Global covariance pooling

import math

plt.switch_backend('agg')



IMAGE_SIZE = 256
BATCH_SIZE = 32 // 2
num_levels = 3
EPOCHS = 100
LR = 0.01
device = torch.device("cuda")
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.95
TRAIN_PRINT_FREQUENCY = 50
EVAL_PRINT_FREQUENCY = 1
#DECAY_EPOCH = [100, 150]

OUTPUT_PATH = 'D:\Jupyter\qiannet'
Loss_list=[]
Accuracy_list = []

# Truncation operation
class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()

        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)

        return output

class SPPLayer(nn.Module):
    def __init__(self, num_levels):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)

            tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                  stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x


# absult value operation
class ABS(nn.Module):
    def __init__(self):
        super(ABS, self).__init__()

    def forward(self, input):
        output = torch.abs(input)
        return output


# add operation
class ADD(nn.Module):
    def __init__(self):
        super(ADD, self).__init__()

    def forward(self, input1, input2):
        output = torch.add(input1, input2)
        return output


class ImageProcessing(nn.Module):
    """Computes convolution with KV filter over the input tensor."""

    def __init__(self) -> None:
        """Constructor"""

        super().__init__()
        # pylint: disable=E1101
        self.kv_filter = (
            torch.tensor(
                [
                    [-1.0, 2.0, -2.0, 2.0, -1.0],
                    [2.0, -6.0, 8.0, -6.0, 2.0],
                    [-2.0, 8.0, -12.0, 8.0, -2.0],
                    [2.0, -6.0, 8.0, -6.0, 2.0],
                    [-1.0, 2.0, -2.0, 2.0, -1.0],
                ], device=torch.device('cuda')
            ).view(1, 1, 5, 5)
            / 12.0
        )  # pylint: enable=E1101

    def forward(self, inp: Tensor) -> Tensor:
        """Returns tensor convolved with KV filter"""

        return F.conv2d(inp.to(torch.float32).to('cuda'), self.kv_filter)
import torch.nn as nn
import torch

class ConvPool(nn.Module):
    """This class returns building block for GNCNN class."""

    def __init__(self, in_channels: int = 16, kernel_size: int = 3, pool_padding: int = 0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=kernel_size, stride=1, padding=0, bias=True)
        self.in_channels = in_channels
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=pool_padding)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Returns conv->gaussian->average pooling."""
        return self.pool(self.gaussian(self.conv(inp)))

    def gaussian(self, x):
        padding, groups, dilation = 1, x.shape[1], 1
        return nn.functional.conv2d(x, torch.ones(groups, 1, 3, 3, device=x.device), padding=padding, groups=groups, bias=None, dilation=dilation)

class ConvGaussianPool(nn.Module):
    """This class returns GNCNN model."""

    def __init__(self, in_channels, kernel_size, pool_padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=kernel_size, stride=1, padding=0, bias=True)
        self.in_channels = in_channels
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=pool_padding)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Returns conv->gaussian->average pooling."""
        return self.pool(self.gaussian(self.conv(inp)))
class GNCNN(nn.Module):
    """This class returns GNCNN model."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = ConvPool(in_channels=1, kernel_size=5, pool_padding=1)
        self.layer2 = ConvPool(pool_padding=1)
        self.layer3 = ConvPool()
        self.layer4 = ConvPool()
        self.layer5 = ConvPool(kernel_size=5)
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, image: Tensor) -> Tensor:
        """Returns logit for the given tensor."""
        with torch.no_grad():
            out = ImageProcessing()(image)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.fully_connected(out)
        return out




    


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, device, train_loader, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()

    for i, sample in enumerate(train_loader):

        data_time.update(time.time() - end)

        data, label = sample['data'], sample['label']

        shape = list(data.size())
        data = data.reshape(shape[0] * shape[1], *shape[2:])
        label = label.reshape(-1)

        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()

        end = time.time()

        output = model(data)  # FP

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)

        losses.update(loss.item(), data.size(0))

        loss.backward()  # BP
        optimizer.step()

        batch_time.update(time.time() - end)  # BATCH TIME = BATCH BP+FP
        end = time.time()
        if i == 200:
            loss = losses
            Loss_list.append(loss.avg)


        if i % TRAIN_PRINT_FREQUENCY == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


# Adjust BN estimated value
def adjust_bn_stats(model, device, train_loader):
    model.train()

    with torch.no_grad():
        for sample in train_loader:
            data, label = sample['data'], sample['label']

            shape = list(data.size())
            data = data.reshape(shape[0] * shape[1], *shape[2:])
            label = label.reshape(-1)

            data, label = data.to(device), label.to(device)

            output = model(data)


def evaluate(model, device, eval_loader, epoch, optimizer, best_acc, PARAMS_PATH):
    model.eval()

    test_loss = 0.0
    correct = 0.0

    with torch.no_grad():
        for sample in eval_loader:
            data, label = sample['data'], sample['label']

            shape = list(data.size())
            data = data.reshape(shape[0] * shape[1], *shape[2:])
            label = label.reshape(-1)

            data, label = data.to(device), label.to(device)

            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

    accuracy = correct / (len(eval_loader.dataset) * 2)


    if accuracy > best_acc and epoch > 50:
        best_acc = accuracy
        all_state = {
            'original_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(all_state, PARAMS_PATH)

    logging.info('-' * 8)
    logging.info('Eval accuracy: {:.4f}'.format(accuracy))
    logging.info('Best accuracy:{:.4f}'.format(best_acc))
    logging.info('-' * 8)
    return best_acc, accuracy


# Initialization
def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)
        nn.init.constant_(module.bias.data, val=0)


# Data augmentation
class AugData():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        # Rotation
        rot = random.randint(0, 3)
        data = np.rot90(data, rot, axes=[1, 2]).copy()

        # Mirroring
        if random.random() < 0.5:
            data = np.flip(data, axis=2).copy()

        new_sample = {'data': data, 'label': label}

        return new_sample


class ToTensor():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        data = np.expand_dims(data, axis=1)
        data = data.astype(np.float32)
        # data = data / 255.0

        new_sample = {
            'data': torch.from_numpy(data),
            'label': torch.from_numpy(label).long(),
        }

        return new_sample


class MyDataset(Dataset):
    def __init__(self, DATASET_DIR, partition, transform=None):
        random.seed(1234)

        self.transform = transform

        self.cover_dir = DATASET_DIR + '/cover'
        self.stego_dir = DATASET_DIR + '/stego'

        self.cover_list = [x.split('/')[-1] for x in glob(self.cover_dir + '/*')]
        random.shuffle(self.cover_list)
        
        count = len(self.cover_list)
        train_count = int(0.6 * count)
        valid_count = int(0.2 * count)
        test_count = int(0.2 * count)
        if (partition == 0):
            self.cover_list = self.cover_list[:train_count]
        if (partition == 1):
            self.cover_list = self.cover_list[train_count:train_count + valid_count]
        if (partition == 2):
            self.cover_list = self.cover_list[:test_count]
        if (partition == 3):
            self.cover_list = self.cover_list[:test_count]
        assert len(self.cover_list) != 0, "cover_dir is empty"

    def __len__(self):
        return len(self.cover_list)

    def __getitem__(self, idx):
        file_index = int(idx)

        cover_path = os.path.join(self.cover_dir, self.cover_list[file_index])
        stego_path = os.path.join(self.stego_dir, self.cover_list[file_index])

        cover_data = cv2.imread(cover_path, -1)
        stego_data = cv2.imread(stego_path, -1)

        data = np.stack([cover_data, stego_data])
        label = np.array([0, 1], dtype='int32')

        sample = {'data': data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


def setLogger(log_path, mode='a'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode=mode)
        file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def main(args):
    statePath = args.statePath

    device = torch.device("cuda")

    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_transform = transforms.Compose([
        AugData(),
        ToTensor()
    ])

    eval_transform = transforms.Compose([
        ToTensor()
    ])

    TRAIN_DATASET_DIR = args.TRAIN_DIR
    VALID_DATASET_DIR = args.VALID_DIR
    TEST_DATASET_DIR = args.TEST_DIR
    TARGET_DATASET_DIR = args.TARGET_DIR

    # Log files
    # Log files
    global Model_NAME
    Model_NAME = 'Qiannet'
    Model_info = '/' + Model_NAME + '/'
    PARAMS_NAME = 'params.pt'
    LOG_NAME = 'model_log'
    date = time.gmtime()
    x = str(date.tm_year) + '_' + str(date.tm_mon) + '_' + str(date.tm_mday) + '_' + str(date.tm_hour + 8) + '_' + str(
        date.tm_min) + '_' + str(date.tm_sec)
    try:
      os.mkdir(os.path.join(OUTPUT_PATH + Model_info))
    except OSError as error:
      print("Folder doesn't exists")
      print(os.path.join(OUTPUT_PATH + Model_info + str(x)))
      os.mkdir(os.path.join(OUTPUT_PATH + Model_info + str(x)))



    PARAMS_PATH = os.path.join(OUTPUT_PATH + Model_info, PARAMS_NAME) 
    LOG_PATH = os.path.join(OUTPUT_PATH + Model_info + str(x), LOG_NAME)
    setLogger(LOG_PATH, mode='w')

    # Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    train_dataset = MyDataset(TRAIN_DATASET_DIR, 0, train_transform)
    valid_dataset = MyDataset(VALID_DATASET_DIR, 1, eval_transform)
    test_dataset = MyDataset(TEST_DATASET_DIR, 2, eval_transform)
    target_dataset = MyDataset(TARGET_DATASET_DIR, 3, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    model = GNCNN().to(device)
    summary(model, input_size=(1, 512, 512))
    #model.apply(initWeights)
    
    """
    params = model.parameters()

    params_wd, params_rest = [], []
    for param_item in params:
        if param_item.requires_grad:
            (params_wd if param_item.dim() != 1 else params_rest).append(param_item)

    param_groups = [{'params': params_wd, 'weight_decay': WEIGHT_DECAY},
                    {'params': params_rest}]

    optimizer = optim.SGD(param_groups, lr=LR, momentum=MOMENTUM, weight_decay=0.0001)
    scheduler = lr_scheduler.MultiStepLR(optimizer, DECAY_EPOCH, 0.1)
    """
    optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=LR, momentum=MOMENTUM, weight_decay=0.0001)
    
    if statePath:
        logging.info('-' * 8)
        logging.info('Load state_dict in {}'.format(statePath))
        logging.info('-' * 8)

        all_state = torch.load(statePath)

        original_state = all_state['original_state']
        optimizer_state = all_state['optimizer_state']
        epoch = all_state['epoch']

        model.load_state_dict(original_state)
        optimizer.load_state_dict(optimizer_state)

        startEpoch = epoch + 1

    else:
        startEpoch = 1

    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.2)
    best_acc = 0.0

    for epoch in range(startEpoch, EPOCHS + 1):
        #optimizer.param_groups[0]['lr'] = LR / math.pow((1 + 10 * (epoch - 1) / EPOCHS), 0.75)
        train(model, device, train_loader, optimizer, epoch)
        #scheduler.step()
        if epoch % EVAL_PRINT_FREQUENCY == 0:
            adjust_bn_stats(model, device, train_loader)
            best_acc, accuracy = evaluate(model, device, valid_loader, epoch, optimizer, best_acc, PARAMS_PATH)
            Accuracy_list.append(accuracy*100)
    #print('Accuracy_list', Accuracy_list)
    #print('Loss_list', Loss_list)
    #np.savetxt('/data/u008/GWF/MYNet/chihua/test/0.4w_chihua_Loss_list.txt', Loss_list, fmt="%.4f")
    #np.savetxt('/data/u008/GWF/MYNet/chihua/test/0.4w_chiuha_Accuracy_list.txt', Accuracy_list, fmt="%.4f")
    np.savetxt('D:\Jupyter\qiannet\Qiannet\Loss_list.txt', Loss_list,fmt="%.4f")
    np.savetxt('D:\Jupyter\qiannet\Qiannet\Accuracy_list.txt', Accuracy_list,fmt="%.4f")

    x1 = range(0, EPOCHS)
    x2 = range(0, EPOCHS)
    y1 = Accuracy_list
    y2 = Loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, '-')
    plt.title('Epoch')
    plt.ylabel('Eval accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '-')
    plt.xlabel('Epoch')
    plt.ylabel('Training loss')
    plt.show()
    plt.savefig("0.3_w_s_100_3_Accuracy_loss.jpg")



    logging.info('\nTest set accuracy: \n')

    # Load best network parmater to test
    all_state = torch.load(PARAMS_PATH)
    original_state = all_state['original_state']
    optimizer_state = all_state['optimizer_state']
    model.load_state_dict(original_state)
    optimizer.load_state_dict(optimizer_state)

    adjust_bn_stats(model, device, train_loader)
    print('source test:')
    evaluate(model, device, test_loader, epoch, optimizer, best_acc, PARAMS_PATH)
    print('target test:')
    evaluate(model, device, target_loader, epoch, optimizer, best_acc, PARAMS_PATH)


def myParseArgs(debug_bool):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-TRAIN_DIR',
        '--TRAIN_DIR',
        help='The path to load train_dataset',
        type=str,
        required=True
    )

    parser.add_argument(
        '-VALID_DIR',
        '--VALID_DIR',
        help='The path to load valid_dataset',
        type=str,
        required=True
    )

    parser.add_argument(
        '-TEST_DIR',
        '--TEST_DIR',
        help='The path to load test_dataset',
        type=str,
        required=True
    )
    parser.add_argument(
        '-TARGET_DIR',
        '--TARGET_DIR',
        help='The path to load target_dataset',
        type=str,
        required=True
    )

    parser.add_argument(
        '-g',
        '--gpuNum',
        help='Determine which gpu to use',
        type=str,
        choices=['0', '1', '2', '3','4','5'],
        required=True
    )

    parser.add_argument(
        '-l',
        '--statePath',
        help='Path for loading model state',
        type=str,
        default=''
    )
    if debug_bool:
        args=parser
    else:
        args = parser.parse_args()

    return args


if __name__ == '__main__':
    #args = myParseArgs()
    debug_bool = True
    #TEST_DIR = '/home/gli/workspace_DL/chy/data/bossbase_dataset/wow_0.1_256'
    #VALID_DIR = '/home/gli/workspace_DL/chy/data/bossbase_dataset/wow_0.1_256'
    #TRAIN_DIR = '/home/gli/workspace_DL/chy/data/bossbase_dataset/wow_0.1_256'
    #TARGET_DIR = '/home/gli/workspace_DL/chy/data/bossbase_dataset/HILL_0.1_256'
    #TRAIN_DIR = '/home/gli/workspace_DL/chy/data/bossbase_dataset/wow_0.4_256'
    TRAIN_DIR = 'D:\stego2'
    #TRAIN_DIR = '/home/gli/workspace_DL/chy/data/bossbase_dataset/wow_0.2_256'
    #TRAIN_DIR = '/home/gli/workspace_DL/chy/data/bossbase_dataset/wow_0.1_256'
    #TRAIN_DIR = '/home/gli/workspace_DL/chy/data/bossbase_dataset/SUNIWARD_0.1_256'
    #TRAIN_DIR = '/home/gli/workspace_DL/chy/data/bossbase_dataset/SUNIWARD_0.2_256'
    TARGET_DIR = 'D:\stego2'
    #TRAIN_DIR = '/home/gli/workspace_DL/chy/data/bossbase_dataset/SUNIWARD_0.4_256'
    #TRAIN_DIR = '/home/gli/workspace_DL/chy/data/bossbase_dataset/SUNIWARD_0.5_256'
    #TRAIN_DIR ='/data/u008/GWF/dataset/HILL_0.4_256'
    #TRAIN_DIR ='/data/u008/GWF/dataset/HILL_0.3_256'
    #TRAIN_DIR ='/data/u008/GWF/dataset/HILL_0.2_256'
    #TRAIN_DIR ='/data/u008/GWF/dataset/HILL_0.1_256'
    #TRAIN_DIR = '/data/u008/GWF/dataset/wow_0.4_256'
    #TRAIN_DIR = '/data/u008/GWF/dataset/SUNIWARD_0.2_256'
    #tst_base_name = '/data/LIRMM/Steganalysis_project/HD_Bases/TST_100k_ASeed123'
    args = myParseArgs(debug_bool=debug_bool)
    if debug_bool:
        args.statePath = None
        #args.statePath = '/data/u008/GWF/Yednet/Yed2/test/0.4w_Yedmodel_params.pt'
        args.gpuNum = '0'  # The reference number of the GPU in the system
        args.TRAIN_DIR = TRAIN_DIR
        args.VALID_DIR = TRAIN_DIR
        args.TEST_DIR = TRAIN_DIR
        args.TARGET_DIR = TARGET_DIR
        #args.test_dir = tst_base_name
        #args.model_name = 'JUNIWARD_P01
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuNum
    main(args)
