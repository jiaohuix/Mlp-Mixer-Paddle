import os
import torch
import paddle
from models import mixer_b16_224_in21k,mixer_b16_224_in1k
from PIL import Image
import matplotlib.pyplot as plt
from paddle.vision import transforms

# 加载路径
path21k='C:/Users/Username/.cache/torch/hub/checkpoints/jx_mixer_b16_224_in21k-617b3de2.pth' #21k
path1k='C:/Users/Username/.cache/torch/hub/checkpoints/mixer_b16_224_miil.pth' # 1k
# 保存路径
save_dir='ckpt'
weights_name21k='mixer_b16_224_in21k.pdparams'
weights_name1k='mixer_b16_224_miil_in1k.pdparams'

def align(load_path,num_classes=21843):
    assert num_classes in [21843,1000]
    map_fn={21843:mixer_b16_224_in21k,1000:mixer_b16_224_in1k}
    torch_keys = []
    paddle_keys=[]
    # 1.1获取torch权重和键值
    torch_weights = torch.load(load_path)
    for k in torch_weights: torch_keys.append(k)
    # 1.2获取paddle权重和键值
    model=map_fn[num_classes](pretrained=False,num_classes=num_classes)
    paddle_weights=model.state_dict()
    for k in paddle_weights:paddle_keys.append(k)

    # 2 将torch权重赋给paddle
    # 有fc,norm,head,proj四类，各自都有bias和weight。其中proj是卷积的参数不变，norm的weight和bias都是一维的不变，fc和head的weight二维，需要转置
    key_pair_length = min(len(torch_keys), len(paddle_keys)) # 获取最小对应权重长度
    for i, k in enumerate(paddle_keys):
        if i >= key_pair_length:
            break
        torch_w=torch_weights[k].detach().numpy()
        if paddle_weights[k].shape == list(torch_w.shape): # paddle shape是list，numpy 的shape是tuple
            paddle_weights[k] = torch_w
            print('b')
        elif paddle_weights[k].shape==list(torch_w.transpose().shape) and k.find('weight') != -1 : # 形状不一致，维度一致，且都是weight
            paddle_weights[k] = torch_w.transpose()
            print('w')
    return paddle_weights


# 对齐权重
# paddle_weights=align(path21k)
# paddle_weights=align(path,num_classes=1000)


# 保存weights
# if not os.path.exists(save_dir):os.makedirs(save_dir)
# paddle.save(paddle_weights,os.path.join(save_dir,weights_name))

# 直接加载权重
# paddle_weights21k=paddle.load(os.path.join(save_dir,weights_name21k))
# paddle_weights1k=paddle.load(os.path.join(save_dir,weights_name1k))
# # model=mixer_b16_224_in21k(num_classes=21843)
# # model.set_dict(paddle_weights21k)
# model=mixer_b16_224_in1k(num_classes=1000)
# model.set_dict(paddle_weights1k)

model=mixer_b16_224_in1k(pretrained=True,num_classes=1000)


# 验证
img=Image.open('./img/cat.9.jpg')
plt.imshow(img)
plt.show()
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
x=transform(img).unsqueeze(0)
model.eval()
logits = model(x)
print(logits)
prob = paddle.nn.Softmax(axis=1)(logits)
pred = paddle.argmax(prob, axis=1)
print(pred)

