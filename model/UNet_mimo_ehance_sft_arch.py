import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.arch_util as arch_util


def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


class PixelShufflePack(nn.Module):
    """Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CABlock(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CABlock, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class SDFN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SDFN, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.max_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.conv_mid = nn.Sequential(
            nn.Conv2d(out_channel * 2, out_channel, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.Sigmoid()
        )

        self.conv_last = nn.Conv2d(out_channel, out_channel, 3, 1, 1)


    def forward(self, x):
        feat1 = self.conv_first(x)
        
        max_pool_feat = self.max_pool(feat1)
        avg_pool_feat = self.avg_pool(feat1)

        attn_map = self.conv_mid(torch.cat((max_pool_feat, avg_pool_feat), dim=1))

        feat2 = feat1 * attn_map
        feat2 = self.conv_last(feat2)
        out = feat1 + feat2 + x
        return out



class HDRUNet_mimo_ehance_sft(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='relu'):
        super(HDRUNet_mimo_ehance_sft, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        
        self.SFT_layer1 = arch_util.SFTLayer()
        self.HR_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.down_conv1 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down_conv2 = nn.Conv2d(nf, nf, 3, 2, 1)

        self.down_conv3 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(nf, nf, 3, 2, 1))

        self.upsample_conv = PixelShufflePack(nf, nf, scale_factor=4, upsample_kernel=3)
        
        basic_block = functools.partial(arch_util.ResBlock_with_SFT, nf=nf)
        self.recon_trunk1 = arch_util.make_layer(basic_block, 2)
        self.recon_trunk2 = arch_util.make_layer(basic_block, 8)
        self.recon_trunk3 = arch_util.make_layer(basic_block, 2)

        self.up_conv1 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        self.up_conv2 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))

        self.SFT_layer2 = arch_util.SFTLayer()
        self.HR_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        cond_in_nc=3
        cond_nf=64
        self.cond_first = nn.Sequential(nn.Conv2d(cond_in_nc, cond_nf, 3, 1, 1), nn.LeakyReLU(0.1, True), 
                                        nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True), 
                                        nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True))
        self.CondNet1 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(cond_nf, 32, 1))
        self.CondNet2 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(cond_nf, 32, 1))
        self.CondNet3 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(cond_nf, 32, 3, 2, 1))

        self.mask_est = nn.Sequential(nn.Conv2d(in_nc, nf, 3, 1, 1), 
                                      nn.ReLU(inplace=True), 
                                      nn.Conv2d(nf, nf, 3, 1, 1),
                                      nn.ReLU(inplace=True), 
                                      nn.Conv2d(nf, nf, 1),
                                      nn.ReLU(inplace=True), 
                                      nn.Conv2d(nf, out_nc, 1))

        self.enhance_block1 = CABlock(nf, kernel_size=3, reduction=4, bias=False, act=nn.PReLU())
        self.sft_block1 = SDFN(in_channel=nf, out_channel=nf)
        self.sft_block2 = SDFN(in_channel=nf, out_channel=nf)

        # activation function
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def check_img_size(self, img):
        b, c, h, w = img.size()

        pad = 4
        pad_h = (pad - h % pad) % pad
        pad_w = (pad - w % pad) % pad

        img = img.view(-1, c, h, w)
        img = F.pad(img, [0, pad_w, 0, pad_h], mode='reflect')
        return img.view(b, c, h + pad_h, w + pad_w)


    def forward(self, x):
        if not isinstance(x, list):
            x = [x, x.clone()]
        mask = self.mask_est(x[0])

        cond = self.cond_first(x[1])   
        cond1 = self.CondNet1(cond)
        cond2 = self.CondNet2(cond)
        cond3 = self.CondNet3(cond)

        fea0 = self.act(self.conv_first(x[0]))
        fea0 = self.SFT_layer1((fea0, cond1))
        fea0 = self.act(self.HR_conv1(fea0))
        fea0_down = self.down_conv3(fea0)

        fea1 = self.act(self.down_conv1(fea0))
        fea1, _ = self.recon_trunk1((fea1, cond2))

        fea2 = self.act(self.down_conv2(fea1))

        out, _ = self.recon_trunk2((fea2, cond3))
        out = out + fea2 + fea0_down  # op1 
        out = self.sft_block1(out)

        out = self.enhance_block1(out) # enhance

        out_upsample = self.upsample_conv(out)  # op2
        out = self.act(self.up_conv1(out)) + fea1

        out, _ = self.recon_trunk3((out, cond2))
        
        out = self.act(self.up_conv2(out)) + fea0 + out_upsample
        out = self.sft_block2(out)

        out = self.SFT_layer2((out, cond1))
        out = self.act(self.HR_conv2(out))
        feat_virual = out.clone()

        out = self.conv_last(out)
        out = mask * x[0] + out
        return out, feat_virual