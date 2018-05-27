import torch
import torch.nn as nn


def Conv2d(in_filters, out_filters, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1)):
    return nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=padding)


def Max2d(kernel_size=(2, 2), stride=2):
    return nn.MaxPool2d(kernel_size=kernel_size, stride=stride)


def Upsample(scale=2):
    return nn.Upsample(scale_factor=scale, mode='nearest')


def UpConv2d(in_filters, out_filters, kernel_size=(3, 3), stride=(2, 2)):
    """Upsample the image by a scale of 2"""
    return nn.ConvTranspose2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=(1, 1),
                              output_padding=(1, 1))


def DownBlock(in_filters, out_filters):
    return nn.Sequential(
        Conv2d(in_filters, out_filters),
        nn.BatchNorm3d(out_filters),
        Conv2d(out_filters, out_filters),
        nn.BatchNorm3d(out_filters),
        Max2d())


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.padding = nn.ReplicationPad2d(1)

        self.downsample = nn.Sequential(
            Conv2d(3, 16),
            nn.BatchNorm3d(16),
            Conv2d(16, 16),
            nn.BatchNorm3d(16),
            Max2d(),  # by 2

            Conv2d(16, 32),
            nn.BatchNorm3d(32),
            Max2d(),  # by 4

            Conv2d(32, 64),
            nn.BatchNorm3d(64),
            Max2d(),  # by 8

            Conv2d(64, 64),
            nn.BatchNorm3d(64),
            Max2d(),  # by 16
            nn.ReLU())

        self.upsample = nn.Sequential(
            UpConv2d(64, 64),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            UpConv2d(64, 32),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            UpConv2d(32, 16),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            UpConv2d(16, 16),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.processing = nn.Sequential(
            # input is assumed to be four channels,
            # processed from features and further input
            Conv2d(19, 16),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            Conv2d(16, 16),
            nn.BatchNorm3d(16),
            nn.ReLU())
        self.final_conv = nn.Sequential(Conv2d(16, 1))

        # size would now be (h/16,w/16)

    def forward(self, in_):
        #
        out_ = self.downsample(in_)
        out_ = self.upsample(out_)

        out_ = torch.cat((out_, in_), 1)
        out_ = self.processing(out_)

        # out_ = self.padding(out_)
        out_ = self.final_conv(out_)
        return out_.squeeze()
