import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
 2/+11   """

    # 初始化层：输入feature，输出feature，权重，偏移
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        # 输入特征，每个输入样本的大小
        self.in_features = in_features
        # 输出特征，每个输出样本的大小
        self.out_features = out_features
        '''
            首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面，所以经过类型转换这个
            self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其m值以达到最优化。
        '''
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        # 偏置，如果设为False，则层将不会学习加法偏差，默认值：True
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))  # Parameters与register_parameter都会向parameters写入参数，但是后者可以支持字符串命名
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # 参数重置函数

    # 初始化权重
    def reset_parameters(self):
        # size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数  size（1）为行，size(1)是指out_features, stdv=1/根号(out_features)
        stdv = 1. / math.sqrt(self.weight.size(1))
        # weight在区间(-stdv, stdv)之间均匀分布随机初始化
        self.weight.data.uniform_(-stdv, stdv)  # uniform()方法将随机生成下一个实数
        # bias均匀分布随机初始化
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    '''
        前馈运算 即计算A~ X W(0) input X与权重W相乘，然后adj矩阵与他们的积稀疏乘直接输入与权重之间进行torch.mm操作，得到support，即XW,
        support与adj进行torch.spmm操作，得到output，即AXW选择是否加bias
    '''
    def forward(self, input, adj):
        # 是矩阵a和b矩阵相乘，torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等
        support = torch.mm(input, self.weight)
        # spmm是稀疏矩阵乘法，减小运算复杂度
        output = torch.spmm(adj, support)
        if self.bias is not None:
            # 返回：系数*输入*权重+偏置
            return output + self.bias
        else:
            # 返回：系数*输入*权重，无偏置
            return output

    def __repr__(self):
        # 打印形式是：GraphConvolution (输入特征 -> 输出特征)
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
