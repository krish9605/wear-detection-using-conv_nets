## imports!
import torch
import torch.nn as nn
import torch.utils.data as data
import os
import os.path as osp
from torchvision import transforms
from torchvision.transforms import functional as TF
import random
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from preprocess import process, Imdb
import pdb
import cv2
from PIL import Image


def imshow(img):
    plt.imshow(img)
    plt.show()


def metric1(y_pred, y_true):
    """ absolute error """
    y_pred = y_pred.squeeze()
    return np.mean(1 - abs((y_pred.data.cpu().numpy() - y_true.data.cpu().numpy())))


def train(train_loader, model, criterion, optimizer, epoch, logger, IS_CUDA=False):
    losses = AverageMeter()
    precision = AverageMeter()
    model.train()
    n_batches = len(train_loader)
    metric_sig = nn.Sigmoid()

    for i_batch, (input, target) in enumerate(train_loader):
        input = input.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)
        input_var = torch.autograd.Variable(input)

        if (IS_CUDA):
            target = target.cuda(async=True)
            input_var = input_var.cuda()

        y_pred = model.forward(input_var)

        if (IS_CUDA):
            y_pred = y_pred.cuda()

        target_var = torch.autograd.Variable(target)

        loss = criterion(y_pred, target_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        m = metric1(metric_sig(y_pred), target_var)
        precision.update(m, input.size(0))
        losses.update(loss.data[0], input.size(0))

    logger.scalar_summary('epoch_loss', losses.avg, epoch)
    print('Epoch: {0}\t, Average_Train_Loss: {1:.4f}, Average_Train_Precision: {2:.2f}'.format(epoch, losses.avg,
                                                                                               precision.avg))


def validate(test_loader, model, epoch, logger, IS_CUDA=False):
    # log.scalar_summary('tp', 0, 0)
    precision = AverageMeter()
    model.eval()
    metric_sig = nn.Sigmoid()

    for i_batch, (input, target) in enumerate(test_loader):
        input = input.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)

        input_var = torch.autograd.Variable(input)
        if (IS_CUDA):
            target = target.cuda(async=True)
            input_var = input_var.cuda()

        target_var = torch.autograd.Variable(target)
        y_pred = model.forward(input_var)

        m = metric1(metric_sig(y_pred), target_var)
        precision.update(m, input.size(0))

    print('VAL\nEpoch: {0}\t Val Precision: {1:.2f}\nVAL'.format(epoch, precision.avg))
    logger.scalar_summary('val_precision', precision.avg, epoch)


## called per image
def process_test_input(input):
    return torch.from_numpy(np.expand_dims(np.expand_dims(input, axis=0), axis=0))


def test(test_image_data, model, IS_CUDA=False):
    precision = AverageMeter()
    model.eval()
    sig_test = nn.Sigmoid()
    print("WARNING: applying sigmoid on output, Check with model")

    predictions = []
    for i_batch, input in enumerate(test_image_data):
        input = torch.from_numpy(np.expand_dims(input, axis=0))
        input = input.type(torch.FloatTensor)

        input_var = torch.autograd.Variable(input)
        if (IS_CUDA):
            input_var = input_var.cuda()

        y_pred = sig_test(model.forward(input_var))

        predictions.append(y_pred.data.cpu().numpy())

    return predictions


def adjust_learning_rate(optimizer, epoch, decay_rate=0.8, decay_epoch=100):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (decay_rate ** (epoch // decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def convert_to_numpy(prep):
    cutout_np = np.asarray([pr[0] for pr in prep])

    a = 0


def get_test_data(data_dir='../test/', num_dir=1, IM_SIZE=(512, 512), threshold=0.5):
    _, _, preprocessed_cutouts, original_img_shape = process(crop_size=IM_SIZE[0], data_dir=data_dir, num_dir=num_dir,
                                                             threshold=threshold)  # preprocessed cutouts contains list of list of cutouts per image

    prep_np = []
    for folder_no in range(len(preprocessed_cutouts)):
        for crop_no in range(len(preprocessed_cutouts[folder_no])):
            preprocessed_cutouts[folder_no][crop_no][0] = \
                preprocessed_cutouts[folder_no][crop_no][0].transpose([2, 0, 1])
        cutout_np = np.asarray([pr[0] for pr in preprocessed_cutouts[folder_no]])

        prep_np.append(cutout_np)

    return prep_np, preprocessed_cutouts, original_img_shape


def get_trainval_data(batch_size, train_percent, num_workers=1, data_dir='../data/', num_dir=1, IM_SIZE=(160, 160),
                      target_im_size=(128, 128), threshold=0.5, transform=True, no_cut_select=0.4):
    wear_cut, no_wear_cut, _, _ = process(crop_size=IM_SIZE[0], data_dir=data_dir, num_dir=num_dir, threshold=threshold)
    images, labels = Imdb(wear_cut, no_wear_cut, no_cut_select=no_cut_select)

    train_idx = random.sample(range(0, len(images)), int(len(images) * train_percent))

    mask = np.zeros(len(images), dtype=bool)
    mask[train_idx] = True

    train_images = np.asarray(images)[mask]
    train_labels = np.asarray(labels)[mask]

    val_images = np.asarray(images)[~mask]
    val_labels = np.asarray(labels)[~mask]

    train_dataset = DatasetTrainVal(train_images, train_labels, target_im_size=target_im_size,
                                    transform=transform)  # TODO: initialise
    val_dataset = DatasetTrainVal(val_images, val_labels, target_im_size=target_im_size,
                                  transform=transform)  # TODO: initialise

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                       num_workers=num_workers)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                     num_workers=num_workers)

    return train_dataloader, val_dataloader


# input -> 128/128
# down -> /2
# down -> /2
# conv
# upsample *2
# upsample *2
#
def process_test(img_shape, imagetest_img_data_locations, predictions, crop_size):
    # heat mapping for one image
    height, width = img_shape[0], img_shape[1]
    desired_size = (crop_size * ((height) // crop_size), crop_size * ((width) // crop_size))
    heat_map = np.zeros(shape=desired_size)
    for cut_locations, pred in zip(imagetest_img_data_locations, predictions):
        x1, x2, y1, y2 = cut_locations
        heat_map[y1:y2, x1:x2] = pred * 255
    return cv2.resize(heat_map, (img_shape[1], img_shape[0]))


class DatasetTrainVal(data.Dataset):
    def __init__(self, input_im, target, target_im_size, transform=None):
        # input data to this is a list of inputs and targets
        self.input = input_im
        self.target = target
        self.target_im_size = target_im_size  # (tuple)
        self.length = self.__len__()
        self.transform_ = transform
        self.original_size = input_im.shape[2]

        if transform:
            self.input = DatasetTrainVal.convert_to_pil(self.input, 1)
            self.target = DatasetTrainVal.convert_to_pil(self.target, 0)
            self.image_template = Image.fromarray(np.zeros([self.original_size, self.original_size, 3], dtype=np.uint8),
                                                  'RGB')

    def __getitem__(self, index):
        image = self.input[index]
        level = self.target[index]
        if self.transform_:
            return self.transform(image, level)
        else:
            return image, level

    def transform(self, image, mask):
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(self.image_template, output_size=self.target_im_size)
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        k = np.array(mask, dtype=np.uint8)

        return np.array(image, dtype=np.int32).transpose([2, 0, 1]), np.array(mask, dtype=np.uint8)

    def __len__(self):
        return len(self.input)

    @staticmethod
    def convert_to_pil(input_list, mode):
        """Converts an input list or ndarray into a PIL list
        :param mode: 0 or 1, 0 for B or W and 1 for RGB
        """
        pil_list = []
        for input in input_list:
            if mode == 0:
                pil_list.append(Image.fromarray(input))
            elif mode == 1:
                pil_list.append(Image.fromarray(input.transpose([1, 2, 0]), mode='RGB'))
        return pil_list


class DatasetTest(data.Dataset):
    def __init__(self, vid_data):
        self.vid_data = vid_data
        self.num_classes = 51
        self.length = self.__len__()

    def __getitem__(self, index):
        feature = self.vid_data[index]['features']
        return feature

    def __len__(self):
        return len(self.vid_data)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # if type(val) is not float:
        #     print('error')
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
