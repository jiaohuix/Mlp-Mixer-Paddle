import os
import time
import random
import paddle
import paddle.nn as nn
from tqdm import tqdm
import numpy as np
import paddle.vision.transforms as T
import matplotlib.pyplot as plt
from utils.logger import logger

def prep_loader(conf):
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    mean = list(map(lambda x: x * 255, IMAGENET_DEFAULT_MEAN))
    std = list(map(lambda x: x * 255, IMAGENET_DEFAULT_STD))
    transform1 = T.Compose([T.Resize(size=(224, 224)),
                            T.RandomHorizontalFlip(0.5),
                            T.Transpose(order=(2, 0, 1)),
                            T.Normalize(mean=mean, std=std)])
    transform2 = T.Compose([T.Resize(size=((224, 224))),
                            T.Transpose(order=(2, 0, 1)),
                            T.Normalize(mean=mean, std=std)])

    # 加载数据
    train_dataset = paddle.vision.datasets.Cifar10(mode='train', transform=transform1)
    dev_dataset = paddle.vision.datasets.Cifar10(mode='test', transform=transform2)  # 验证集使用与训练集相同的增强策略，检验模型的泛化能力
    # 加载dataloader
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=conf['hparas']['batch_size'], shuffle=True)
    dev_loader = paddle.io.DataLoader(dev_dataset, batch_size=conf['hparas']['batch_size'], shuffle=False)
    return train_loader,dev_loader

def same_seeds(seed=1024):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def draw_process(title,color,iters,data,label):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel(label, fontsize=20)
    plt.plot(iters, data,color=color,label=label)
    plt.legend()
    plt.grid()
    plt.show()

def prep_model(conf,model):
    criterion = paddle.nn.CrossEntropyLoss()
    lr=eval(conf['hparas']['learning_strategy']['lr'])
    optimizer = paddle.optimizer.Adam(learning_rate=lr,parameters=model.parameters())
    metric = paddle.metric.Accuracy()
    return criterion,optimizer,metric

def train(conf,model,train_loader,dev_loader):
    model.train()
    criterion, optimizer,metric=prep_model(conf,model)
    steps = 0
    Iters, total_loss, total_acc = [], [], []
    max_epochs=conf['hparas']['num_epochs']
    total_steps=len(train_loader)*max_epochs
    tic_train = time.time()
    for epoch in range(max_epochs):
        for _, batch in enumerate(tqdm(train_loader)):
            steps += 1
            # forward
            imgs,labels=batch
            logits = model(imgs)
            loss = criterion(logits, labels)
            # acc
            correct = metric.compute(logits, labels)
            metric.update(correct)
            acc = metric.accumulate()
            # backward
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if steps % conf['hparas']["log_steps"] == 0:
                Iters.append(steps)
                total_loss.append(float(loss))
                total_acc.append(float(acc))
                #打印中间过程
                logger.info(f"Train | Epoch: [{epoch+1}/{max_epochs}] | Step: [{steps}/{total_steps}]"
                      f" Loss: {float(loss):.4f} | Acc: {float(acc):.4f} | Speed:{conf['hparas']['log_steps']/(time.time()-tic_train):.2f} step/s")
                tic_train = time.time()

            #保存模型参数
            if steps % conf['hparas']["save_steps"] == 0:
                save_path = os.path.join(conf['hparas']["save_dir"],'model_{}.pdparams'.format(steps))
                logger.info(f'Train | Save model to: ' + save_path)
                paddle.save(model.state_dict(),save_path)
            # 评估模型
            if steps % conf['hparas']["val_steps"] == 0:
                evaluate(model,dev_loader)
        metric.reset()
    paddle.save(model.state_dict(),os.path.join(conf['hparas']["save_dir"],"final.pdparams"))
    draw_process("trainning loss","red",Iters,total_loss,"trainning loss")
    draw_process("trainning acc","green",Iters,total_acc,"trainning acc")

@paddle.no_grad()
def evaluate(model,data_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss,correct=0,0
    for batch in tqdm(data_loader):
        imgs,labels=batch
        logits=model(imgs)
        batch_loss = criterion(logits, labels)
        loss+=float(batch_loss)
        pred=paddle.argmax(logits,axis=1).numpy()
        labels=labels.numpy().reshape(-1)
        correct+=np.sum(pred==labels)
    loss=loss/len(data_loader)
    acc=correct/len(data_loader.dataset)
    logger.info(f'Eval | Loss: {loss:.4f} | Acc: {acc:.4f}')
    model.train()