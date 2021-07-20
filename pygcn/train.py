from __future__ import division
from __future__ import print_function
import os
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

# 将变量以标签-值的字典形式存入args字典
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# 在神经网络中，参数默认是进行随机初始化的。如果不设置随机种子的话每次训练时的初始化都是随机的，导致结果不确定。如果设置初始化，则每次初始化都是固定的
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# 自动选择是否用cuda
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data 这里可以在jupyter中看到结果
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer 调用模型：下面几个参数在models.py中声明
'''
    第一个参数为底层节点的参数，feature的个数
    nhid，隐层节点个数
    nclass，最终的分类数
    dropout正则化化放置过拟合，在每一层的神经网层中以一定概率丢弃神经元
'''
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# 将数据写入cuda便于后续加速
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()  # 返回当前时间
    model.train()
    optimizer.zero_grad()  # 意思是把梯度置零，也就是把loss关于weight的导数变成0，pytorch中每一轮batch需要设置optimizer.zero_gra
    output = model(features, adj)  # 前向传播
    '''
        由于在算output时已经使用了log_softmax，这里使用的损失函数就是NLLloss，如果前面没有加log运算，这里就要使用CrossEntropyLoss了,损失函数NLLLoss() 的输入是
        一个对数概率向量和一个目标标签. 它不会为我们计算对数概率，适合最后一层是log_softmax()的网络. 损失函数 CrossEntropyLoss() 与 NLLLoss() 类似,
        唯一的不同是它为我们去做 softmax.可以理解为：CrossEntropyLoss()=log_softmax() + NLLLoss()
    '''
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()  # 计算准确率
    optimizer.step()  # 反向求导，也就是梯度下降，更新值

    # 如果不在训练期间进行验证
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        # model.eval()固定语句，主要针对不启用BatchNormalization和Dropout
        model.eval()  # 函数用来执行一个字符串表达式，并返回表达式的值
        output = model(features, adj)  # 前向传播

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

# 定义测试函数，相当于对已有的模型在测试集上运行对应的loss与accuracy
def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model 逐个epoch进行train，最后test
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
