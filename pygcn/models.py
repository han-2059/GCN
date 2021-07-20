import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution


class GCN(nn.Module):
    # 底层节点的参数，feature的个数；隐层节点个数；最终的分类数
    def __init__(self, nfeat, nhid, nclass, dropout):
        '''
            :param nfeat: 底层节点的参数，feature的个数
            :param nhid: 隐层节点个数
            :param nclass: 最终的分类数
            :param dropout: dropout参数
        '''
        # super()._init_()在利用父类里的对象构造函数
        super(GCN, self).__init__()
        # self.gc1代表GraphConvolution()，GraphConvolution()这个方法类在layer里面，gc1输入尺寸nfeat，输出尺寸nhid
        self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2代表GraphConvolution()，gc2输入尺寸nhid，输出尺寸ncalss
        self.gc2 = GraphConvolution(nhid, nclass)
        # dropout参数
        self.dropout = dropout

    # 前向传播，按照个人理解，F.log_softmax(x, dim=1)中的参数x就是每一个节点的embedding
    def forward(self, x, adj):
        # 括号里面x是输入特征，adj是邻接矩阵。self.gc1(x, adj)执行GraphConvolution中forward函数
        x = F.relu(self.gc1(x, adj))

        # 输入x，dropout参数是self.dropout。training=self.training表示将模型整体的training状态参数传入dropout函数，没有此参数无法进行dropout
        x = F.dropout(x, self.dropout, training=self.training)

        # gc2层
        x = self.gc2(x, adj)

        # 输出为输出层做log_softmax变换的结果，dim表示log_softmax将计算的维度
        return F.log_softmax(x, dim=1)
