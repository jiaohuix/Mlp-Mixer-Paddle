import math
import paddle
import paddle.nn as nn
from copy import deepcopy
from functools import partial
from .layers import Mlp,DropPath,Identity,PatchEmbed,lecun_normal_,to_2tuple
from paddle.nn.initializer import Constant,Normal,XavierUniform
from .helpers import named_apply
from .registry import register_model

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 0.875, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'stem.proj', 'classifier': 'head',
        **kwargs
    }

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

default_cfgs = dict(
    mixer_s32_224=_cfg(),
    mixer_s16_224=_cfg(),
    mixer_b32_224=_cfg(),
    mixer_b16_224=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224-76587d61.pth',
    ),
    mixer_b16_224_in21k=_cfg(
        url='./ckpt/mixer_b16_224_in21k.pdparams',
        num_classes=21843
    ),
    mixer_b16_224_in1k=_cfg(
        url='./ckpt/mixer_b16_224_miil_in1k.pdparams',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear',
    ),
    mixer_l32_224=_cfg(),
    mixer_l16_224=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_l16_224-92f9adc4.pth',
    ),
    mixer_l16_224_in21k=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_l16_224_in21k-846aa33c.pth',
        num_classes=21843
    ),



)
normal_ = Normal(std=1e-6)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)

class MixerBlock(nn.Layer):#含token和channel
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self, dim, seq_len,  mlp_ratio=(0.5, 4.0),mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, epsilon=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):#[bsz 196 768]
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose((0,2,1))).transpose((0,2,1))) # paddle的transpose轴需要按顺序
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


# 模型结构
class MlpMixer(nn.Layer):
    def __init__(
            self,
            num_classes=1000,
            img_size=224,
            in_chans=3,
            patch_size=16,
            num_blocks=8,
            embed_dim=512,
            mlp_ratio=(0.5, 4.0), # ds和dc的比例（与embed_dim比）
            block_layer=MixerBlock,
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
            act_layer=nn.GELU,
            drop_rate=0.,
            drop_path_rate=0.,
            nlhb=False, # false时head_bias=0
            stem_norm=False,# patch embed是否用原来的norm（layernorm），默认不用norm，即用恒等映射
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.stem = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, norm_layer=norm_layer if stem_norm else None)
        # FIXME drop_path (stochastic depth scaling rule or all the same?)
        self.blocks = nn.Sequential(*[
            block_layer(
                embed_dim, self.stem.num_patches, mlp_ratio, mlp_layer=mlp_layer, norm_layer=norm_layer,
                act_layer=act_layer, drop=drop_rate, drop_path=drop_path_rate)
            for _ in range(num_blocks)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes) if num_classes > 0 else Identity() # 分类头

        self.init_weights(nlhb=nlhb)
        self.apply(self.init_weights)

    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.
        named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # depth-first

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''): # 重置分类头
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else Identity()

    def forward_features(self, x): # 得到最后一维向量
        x = self.stem(x) # 嵌入为向量
        x = self.blocks(x) # n 层mixerlayer
        x = self.norm(x)
        x = x.mean(axis=1)
        return x

    def forward(self, x): #分类
        x = self.forward_features(x)
        x = self.head(x)
        return x

def _init_weights(module: nn.Layer, name: str, head_bias: float = 0., flax=False):
    """ Mixer weight initialization (trying to match Flax defaults)
    """
    xavier_uniform_=XavierUniform()

    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            zeros_(module.weight)
            constant_ = Constant(value=head_bias)
            constant_(module.bias)
        else:
            if flax:
                # Flax defaults
                lecun_normal_(module.weight)
                if module.bias is not None:
                    zeros_(module.bias)
            else:
                # like MLP init in vit (my original init)
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        normal_(module.bias)
                    else:
                        zeros_(module.bias)
    elif isinstance(module, nn.Conv2D):
        lecun_normal_(module.weight)
        if module.bias is not None:
            zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2D, nn.GroupNorm)):
        ones_(module.weight)
        zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        # NOTE if a parent module contains init_weights method, it can override the init of the
        # child modules as this will be called in depth-first order.
        module.init_weights()


def _create_mixer(variant, pretrained=False, **kwargs):
    num_classes=kwargs.get('num_classes', None) # 外面传的类别
    cfg=default_cfgs[variant]
    is_fintuned=True if num_classes is not None and num_classes!=cfg['num_classes'] else False
    if num_classes is None:kwargs.setdefault('num_classes',cfg['num_classes'])
    model = MlpMixer(**kwargs)
    if pretrained and cfg['url'].find('pdparams')!=-1:
        state_dict=paddle.load(cfg['url'])
        if is_fintuned: # 自己传了类别，且与预训练权重的类别不符，修改head的权重和偏置
            state_dict['head.weight']=model.head.weight
            state_dict['head.bias']=model.head.bias
        model.set_dict(state_dict)
    return model


@register_model
def mixer_s32_224(pretrained=False, **kwargs):
    """ Mixer-S/32 224x224
    Paper: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=32, num_blocks=8, embed_dim=512, **kwargs)
    model = _create_mixer('mixer_s32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_s16_224(pretrained=False, **kwargs):
    """ Mixer-S/16 224x224
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512, **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_b32_224(pretrained=False, **kwargs):
    """ Mixer-B/32 224x224
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=32, num_blocks=12, embed_dim=768, **kwargs)
    model = _create_mixer('mixer_b32_224', pretrained=pretrained, **model_args)
    return model

## 模型结构配置
@register_model
def mixer_b16_224(pretrained=False, **kwargs):
    """ Mixer-B/16 224x224. ImageNet-1k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    model = _create_mixer('mixer_b16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_b16_224_in21k(pretrained=False, **kwargs):
    """ Mixer-B/16 224x224. ImageNet-21k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    model = _create_mixer('mixer_b16_224_in21k', pretrained=pretrained, **model_args)
    return model

@register_model
def mixer_b16_224_in1k(pretrained=False, **kwargs):
    """ Mixer-B/16 224x224. ImageNet-1k pretrained weights.
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_args = dict(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    model = _create_mixer('mixer_b16_224_in1k', pretrained=pretrained, **model_args)
    return model

@register_model
def mixer_l32_224(pretrained=False, **kwargs):
    """ Mixer-L/32 224x224.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=32, num_blocks=24, embed_dim=1024, **kwargs)
    model = _create_mixer('mixer_l32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_l16_224(pretrained=False, **kwargs):
    """ Mixer-L/16 224x224. ImageNet-1k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=24, embed_dim=1024, **kwargs)
    model = _create_mixer('mixer_l16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_l16_224_in21k(pretrained=False, **kwargs):
    """ Mixer-L/16 224x224. ImageNet-21k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=24, embed_dim=1024, **kwargs)
    model = _create_mixer('mixer_l16_224_in21k', pretrained=pretrained, **model_args)
    return model


