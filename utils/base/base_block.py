from torch import nn


def encode(in_filters, out_filters, bn=True):
    block = [
        nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout2d(0.1),
    ]
    if bn:
        block.append(nn.BatchNorm2d(out_filters))
    return block


def decode(in_filters, out_filters, bn=True):
    block = [nn.ConvTranspose2d(in_filters, out_filters, 4, 2, 1)]
    if bn:
        block.append(nn.BatchNorm2d(out_filters))
    block.append(nn.LeakyReLU(0.2, inplace=True))
    return block
