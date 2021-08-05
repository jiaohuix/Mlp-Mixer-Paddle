from .mixer_paddle import *
import os
import paddle
import models
from utils import logger

def build_model(conf):
    assert conf['model']['name'] in ['1k','21k']
    map_fn={'1k':'mixer_b16_224_in1k','21k':'mixer_b16_224_in21k'}
    # 加载预训练模型
    model=getattr(models,map_fn[conf['model']['name']])(pretrained=conf['model']['pretrained'],
                                                       num_classes=conf['data']['class_dim'])

    if conf['model']['pretrained']:logger.info('Prep | Pretrained model loaded!')
    # 加载微调模型
    model_path=os.path.join(conf['hparas']["save_dir"],"final.pdparams")
    if os.path.exists(model_path):
        model.set_dict(paddle.load(model_path))
        logger.info('Prep | Fintuned model loaded!')
    # 加载微调的模型
    return model