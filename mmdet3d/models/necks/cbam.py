'''
Description: https://www.yuque.com/huangzhongqing/lxph5a/mur8gs#o8sag
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-10-19 10:47:15
LastEditTime: 2021-10-21 13:00:01
FilePath: /mmdetection3d/mmdet3d/models/necks/cbam.py
'''
import torch
import torch.nn as nn
import torchvision

# 在通道维度上进行全局的pooling操作，再经过同一个MLP得到权重，相加作为最终的注意力向量（权重）。
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 1是输出特征图大小
        self.max_pool = nn.AdaptiveMaxPool2d(1) # 

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False), # nn.Conv2d(输入channels，输出channels，卷积核kernels_size=3, padding=1, bias=False)
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x)) # torch.Size([1, 16, 1, 1])
        print('【Channel】avgout.shape {}'.format(avgout.shape)) # torch.Size([1, 16, 1, 1])
        maxout = self.shared_MLP(self.max_pool(x)) # torch.Size([1, 16, 1, 1])
        return self.sigmoid(avgout + maxout) # torch.Size([1, 16, 1, 1])

# 通道注意力
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3) # 卷积核大小7
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True) # (1,1,64,64)
        maxout, _ = torch.max(x, dim=1, keepdim=True) # # (1,1,64,64)
        out = torch.cat([avgout, maxout], dim=1) # (1,2,64,64)
        out = self.sigmoid(self.conv2d(out)) #  (1,2,64,64)-->(1,1,64,64)     nn.Conv2d(in_channels=2, out_channels=1
        return out

# CBAM调用
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel) # 通道注意力
        self.spatial_attention = SpatialAttentionModule() # 空间注意力

    def forward(self, x): # main func
        out = self.channel_attention(x) * x #通道 torch.Size([1, 16, 1, 1])   *  torch.Size([1, 16, 64, 64])
        print('outchannels:{}'.format(out.shape)) # outchannels:torch.Size([1, 16, 64, 64])
        # out = self.spatial_attention(out) * out #空间   # torch.Size([1, 1, 64, 64])   *  torch.Size([1, 16, 64, 64])
        return out


class ResBlock_CBAM(nn.Module):
    def __init__(self,in_places, places, stride=1,downsampling=False, expansion = 4):
        super(ResBlock_CBAM,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )
        self.cbam = CBAM(channel=places*self.expansion) # 初始化cbam 输入参数channel

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x) 
        print(x.shape) # torch.Size([1, 16, 64, 64])
        out = self.cbam(out) # 调用cbam=============================
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

model = ResBlock_CBAM(in_places=16, places=4) # 调用
print(model)

input = torch.randn(1, 16, 64, 64) # (B C H W)注意维度
out = model(input)
print('out.shape {}'.format(out.shape)) # torch.Size([1, 16, 64, 64])

# 运行代码： python mmdet3d/models/necks/cbam.py
''' 
ResBlock_CBAM(
  (bottleneck): Sequential(
    (0): Conv2d(16, 4, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (4): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): Conv2d(4, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (7): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (cbam): CBAM(
    (channel_attention): ChannelAttentionModule(
      (avg_pool): AdaptiveAvgPool2d(output_size=1)
      (max_pool): AdaptiveMaxPool2d(output_size=1)
      (shared_MLP): Sequential(
        (0): Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): ReLU()
        (2): Conv2d(1, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
      (sigmoid): Sigmoid()
    )
    (spatial_attention): SpatialAttentionModule(
      (conv2d): Conv2d(2, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (sigmoid): Sigmoid()
    )
  )
  (relu): ReLU(inplace=True)
)
torch.Size([1, 16, 64, 64])
[Channel]avgout.shape torch.Size([1, 16, 1, 1])
outchannels:torch.Size([1, 16, 64, 64])
out.shape torch.Size([1, 16, 64, 64])

'''