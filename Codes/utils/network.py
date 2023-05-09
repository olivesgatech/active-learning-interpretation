# Script contains class definition for the convolutional autoencoder-based network architecture used in the paper

import torch.nn as nn

def double_conv_down(in_channels, out_channels):
    """Convolutional block for encoding path"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def double_conv_up(in_channels, out_channels):
    """Convolutional block for decoding path"""
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, padding=0, dilation=1, kernel_size=3, stride=2, output_padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),    
    )

class FaciesSegNet(nn.Module):

    def __init__(self, n_class, out_channels=(8,10,30,40,60)):
        
        super().__init__()
        
        # set up encoder
        self.down_convs = nn.ModuleList([double_conv_down(out_channels[i], out_channels[i+1]) for i in range(len(out_channels)-1)])
        self.down_convs.insert(0, double_conv_down(1, out_channels[0]))
        
        # set up decoder
        self.up_convs = nn.ModuleList([double_conv_up(out_channels[i], out_channels[i-1]) for i in range(len(out_channels)-1, 0, -1)])
        
        # maxpool
        self.maxpool = nn.MaxPool2d(2)

        # last layers
        self.conv_last = nn.Conv2d(out_channels[0], n_class, 1)
        self.conv_reconstruct = nn.Conv2d(out_channels[0], 1, 1)

    def forward(self, section):
        # pass through encoding branch
        x = section
        for i, block in enumerate(self.down_convs):
            x = block(x)
            
            if i < len(self.down_convs) - 1:
                x = self.maxpool(x)
            
        # pass through decoding branch
        for block in self.up_convs:
            x = block(x)

        out = self.conv_last(x)[:, :, :section.shape[2], :section.shape[3]]
        reconstruct = self.conv_reconstruct(x)[:, :, :section.shape[2], :section.shape[3]]

        return out, reconstruct
