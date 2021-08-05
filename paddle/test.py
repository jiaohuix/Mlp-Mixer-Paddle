import paddle
import paddle.nn as nn

# tip1: 初始化的使用
const_=nn.initializer.Constant(value=0)
x=paddle.randn((3,2))
print(x)
const_(x) # 没有返回值
print(x)

# tip2：clamp
# import numpy as np
# mi=0.5
# ma=0.7
# x=paddle.randn((3,2))
# x=x.numpy()
# print(x)
# x=np.where(x>mi,x,mi)
# x=np.where(x<ma,x,ma)
# print(x)

# tip3:named
class mod(nn.Layer):
    def __init__(self):
        super(mod,self).__init__()
        self.linear=nn.Linear(3,2)
        self.linear2=nn.Linear(2,1)

    def forward(self, x):
        x=self.linear(x)
        x=self.linear2(x)
        return x

# module=mod()
# name='323'
# for child_name, child_module in module.named_children():
#     child_name = '.'.join((name, child_name)) if name else child_name
#     print(child_name)

# tim4: kwargs op [pop get setdefault]
def test(**kwargs): #**kwargs把参数打包成字典给函数
    print(kwargs,type(kwargs))
test(a=1,b=2)# {'a': 1, 'b': 2} <class 'dict'>
## get
dic=dict(a=1,b=2)
print(dic)
c=dic.get('c','c') # 友好的访问值的方式，如果key不存在，返回None，不会报错；可以设置默认值
print(c) #c
## setdefault
d=dic.setdefault('d','d') # 添加不存在的键，默认值为None，值可以自己设置
dic.setdefault('b','b') # 不会修改已经存在的键
print(dic) #
print(d) # 返回值是设定的不存在的键的值，若无则为None

## pop
# aval=dic.pop('a') # 弹出指定键的键值对 ,若键不存在会报错
# print(dic) #{'b': 2, 'd': 'd'}
# print(aval) #1

ppp=dic.pop('ppp',False) # pop时若键不存在，必须指定默认值才不会报错；不像get，键不存在且没有默认值时返回None
print(ppp)
# kwargs={'patch_size': 16, 'num_blocks': 12, 'embed_dim': 768, 'num_classes': 21843, 'in_chans': 3, 'img_size': (224, 224)}
# pruned = kwargs.pop('pruned', False)  # false
# print(pruned)