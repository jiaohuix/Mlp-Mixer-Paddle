# Mlp-Mixer论文复现

​		Mlp-mixer是谷歌5月份提出的基于纯mlp结构的cv框架，用来替代卷积和Transformer里的注意力操作。

​		本项目用b16规模的mixer，使用在imagenet和imagenet21k上预训练的权重，在cifar10数据集上验证准确率分别达到了97.1%（1epoch）和97.07%（2epoch）。



## Train

```
python main.py --config ./conf/base.yaml --mode train
或
./run.sh 1
```

## Evaluate

```
python main.py  --config ./conf/base.yaml --mode eval
或
./run.sh 2
```

## Link

**注：换预训练权重需要修改yaml配置文件里的model name；由于没有存放预训练权重，可以移步aistudio直接运行：**

[aistudio Mlp-Mixer Paddle 复现](https://aistudio.baidu.com/aistudio/projectdetail/2258020)

[Mlp-Mixer论文地址](https://arxiv.org/pdf/2105.01601v4.pdf)

[csdn:Mlp-Mixer简介](https://blog.csdn.net/weixin_43312063/article/details/117250816?spm=1001.2014.3001.5501)
