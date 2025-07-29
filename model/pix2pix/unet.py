import numpy as np
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")



def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """

    return  nn.InstanceNorm2d(channels) #GroupNorm32(16, channels)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class Downsample(nn.Module): 
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2) 
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class Upsample(nn.Module): 
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv 
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels), 
            nn.ReLU() if up else nn.LeakyReLU() if down else nn.SiLU(),

        )
        self.in_layers_onepix = nn.ReLU() if up else nn.LeakyReLU() if down else nn.SiLU()

        self.updown = up or down 

        if up:
            self.h_upd = Upsample(channels, self.use_conv, dims)
            self.x_upd = Upsample(channels, self.use_conv, dims)
        elif down:
            self.h_upd = Downsample(channels, self.use_conv, dims)
            self.x_upd = Downsample(channels, self.use_conv, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()



        self.out_layers =conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)


    def forward(self, x): 
        self.in_layers = self.in_layers_onepix if x.shape[-1]==1 else self.in_layers
        if self.updown:

            h = self.in_layers(x)
            h = self.h_upd(h) 
            h = self.out_layers(h)

            x = self.x_upd(x) 
             
        else:
            h = self.out_layers(self.in_layers(x))
  
        return self.skip_connection(x) + h


class UNetModel(nn.Module):

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
  
        use_conv = False, 
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=False, #
        dims=2, # 
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False, #
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,

        use_spatial_transformer=False, 
        legacy=True,
        use_tanh = True
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
   
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        self.input_blocks = nn.ModuleList(
            [
            conv_nd(dims, in_channels, model_channels, 3, padding=1)
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels 
        ds = 1
        for level, mult in enumerate(channel_mult): #level0 1 2 3，mult1 2 4 8
            for _ in range(num_res_blocks): 
                layers = [
                    ResBlock(
                        ch,
                        # time_embed_dim,
                        dropout,
                        use_conv = use_conv,
                        out_channels=mult * model_channels, 
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels 

                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1: 
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(
                        ResBlock(
                            ch,
                            # time_embed_dim,
                            dropout,
                            use_conv = use_conv,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch 
                input_block_chans.append(ch)
                ds *= 2 
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = nn.Sequential(
            ResBlock(
                ch,
   
                dropout,
                dims=dims,
                use_conv = use_conv,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),

            ResBlock(
                ch,
                dropout,
                dims=dims,
                use_conv = use_conv,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]: #8,4,2,1
            for i in range(num_res_blocks + 1): 
                ich = input_block_chans.pop() 
                layers = [
                    ResBlock(
                        ch + ich,
                        dropout,
                        out_channels=model_channels * mult,
                        use_conv = use_conv,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult

                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            dropout,
                            use_conv = use_conv,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2 
                self.output_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.ReLU(),
            conv_nd(dims, model_channels, out_channels, 3, padding=1), 
            nn.Tanh() if use_tanh else nn.Identity(), 
        )


    def forward(self, x):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h) 
            hs.append(h)
        h = self.middle_block(h) 
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h) 
        h = h.type(x.dtype)
        return self.out(h)
    


class SimpleDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(SimpleDiscriminator, self).__init__()

        def discriminator_block_downsize(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1), nn.InstanceNorm2d(out_filters), nn.LeakyReLU(0.2, inplace=True)]
            return layers
        
        def discriminator_block_nodownsize(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1), nn.InstanceNorm2d(out_filters), nn.LeakyReLU(0.2, inplace=True)]
            return layers

        self.model = nn.Sequential(
            *discriminator_block_downsize(in_channels, 64), #256--128
            *discriminator_block_downsize(64, 128), #128--64
            *discriminator_block_downsize(128, 256), #64--32
            *discriminator_block_nodownsize(256, 512), #32--32 
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img_A, img_B): #(bz, channel, 256, 256)
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

