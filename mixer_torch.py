'''
MLP-Mixer代码复现
author:角灰大帝
data:21/5/24
Note:
    1.per-patch fc can be replaced by a conv with kernel_size=stride=patch_size
    2.mlp1 and mlp2 can merge to a block,and has dropout
    3.there is a LayerNorm before global avg pool
    4.avg pool just calculate mean per channel rather than use avgpool1d
    5.classification head has one fc layers
'''
import torch
import torch.nn as nn

class MlpBlock(nn.Module):
    def __init__(self,in_dim,hidden_dim,drop_rate=0):
        super(MlpBlock,self).__init__()
        self.mlp=nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(drop_rate))
    def forward(self,x):
        return self.mlp(x)

class MixerLayer(nn.Module):
    '''
    ns:序列数（patch个数）；nc：通道数（嵌入维度）；ds：token mixing的隐层神经元数；dc：channel mixing的隐层神经元数；
    '''
    def __init__(self,ns,nc,ds=256,dc=2048,drop_rate=0.):
        super(MixerLayer,self).__init__()
        self.norm1=nn.LayerNorm(nc)
        self.norm2=nn.LayerNorm(nc)
        self.tokenMix=MlpBlock(in_dim=ns,hidden_dim=ds,drop_rate=drop_rate)
        self.channelMix=MlpBlock(in_dim=nc,hidden_dim=dc,drop_rate=drop_rate)

    def forward(self,x):
        x=self.norm1(x)
        x2=self.tokenMix(x.transpose(1,2)).transpose(1,2) # 不能用.T,否则b会跑到最后一维
        x=x+x2
        x2=self.norm2(x)
        x2=self.channelMix(x2)
        return x+x2

class Mixer(nn.Module):
    ## paper:paper：https://arxiv.org/pdf/2105.01601.pdf ##
    def __init__(self,num_classes,image_size,patch_size=16,num_layers=8,embed_dim=512,ds=256,dc=2048,drop_rate=0):
        '''
        :param image_size: 输入图像分辨率
        :param num_classes: 分类类别数
        :param num_layers: mixer层数
        :param patch_size: patch的宽高
        :param embed_dim: 通道数C
        :param ds: token-mixing的隐层神经元数
        :param dc: channel-mixing的隐层神经元数
        '''
        super(Mixer,self).__init__()
        assert image_size%patch_size==0
        self.embed = nn.Conv2d(3,embed_dim,kernel_size=patch_size,stride=patch_size)
        ns=(image_size//patch_size)**2 # 序列数
        MixBlock=MixerLayer(ns=ns,nc=embed_dim,ds=ds,dc=dc,drop_rate=drop_rate)
        self.mixlayers=nn.Sequential(*[MixBlock for _ in range(num_layers)])
        self.norm=nn.LayerNorm(embed_dim)
        self.cls=nn.Linear(embed_dim,num_classes)

    def forward(self,x):
        x=self.embed(x).flatten(2).transpose(1,2) # n c2 hw->n hw c2
        x=self.mixlayers(x)
        x=self.norm(x)
        x=torch.mean(x,dim=1) # 逐通道求均值 N C
        x=self.cls(x)
        return x

if __name__ == '__main__':
    ###test mixer###
    x = torch.randn(2, 3,224, 224)
    model = Mixer(3,image_size=224)
    out = model(x)
    print(out.shape)