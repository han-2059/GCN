import numpy as np
import scipy.sparse as sp
import torch

'''
    先将所有由字符串表示的标签数组用set保存，set的重要特征就是元素没有重复，
    因此表示成set后可以直接得到所有标签的总数，随后为每个标签分配一个编号，创建一个单位矩阵，
    单位矩阵的每一行对应一个one-hot向量，也就是np.identity(len(classes))[i, :]，
    再将每个数据对应的标签表示成的one-hot向量，类型为numpy数组
'''
def encode_onehot(labels):
    classes = set(labels)  # set()函数创建一个无序不重复元素集
    # identity创建一个n*n的矩阵
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}

    # get函数得到字典key对应的value值，字典key为label的值，value为矩阵的每一行，enumerate函数用于将一个可遍历的数据对象组合为一个索引序列
    # map() 会根据提供的函数对指定序列做映射，第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    # content file的每一行的格式为: <paper_id> <word_attributes> <class_label>，分别对应 0, 1:-1, -1
    # feature为第二列到倒数第二列，labels为最后一列
    # idx_features_labels将cora数据集中的数据读取出来
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),dtype=np.dtype(str))
    # 存储csr型稀疏矩阵，idx_features_labels[:, 1:-1]表示取所有行，第一列到倒数第二列
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # idx_features_labels[:, -1]表示取所有行，最后一列
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    # cites file的每一行格式为: <cited paper ID>  <citing paper ID>
    # 根据前面的contents与这里的cites创建图，算出edges矩阵与adj矩阵，这里是取出数据集中的第一列，也就是<cited paper ID>
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

    # 由于文件中节点并非是按顺序排列的，因此建立一个编号为0-(node_size-1)的哈希表idx_map，
    # 哈希表中每一项为old id: number，即节点id对应的编号为number
    # 这行代码将第一列数据做了这样的操作：j表示的是值，i表示的是值对应的索引，以<j,i>形式存储数据，然后根据j的值进行从小到大为<j,i>排序
    idx_map = {j: i for i, j in enumerate(idx)}

    # edges_unordered为直接从cora.cites文件中直接读取的结果，是一个(edge_num, 2)的数组，每一行表示一条边两个端点的idx
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)

    # flatten：降维，返回一维数组，这里就是将edges_unordered这个（edge_num, 2）维的矩阵变成一个一维数组
    # 边的edges_unordered中存储的是端点id，要将每一项的old id换成编号number
    # 在idx_map中以idx作为键查找得到对应节点的编号（或者说对应的索引），reshape生成与edges_unordered形状一样的数组
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

    # 根据coo矩阵性质，这一段的作用就是，网络有多少条边，邻接矩阵就有多少个1，
    # 所以先创建一个长度为edge_num的全1数组，每个1的填充位置就是一条边中两个端点的编号，即edges[:, 0], edges[:, 1]，矩阵的形状为(node_size, node_size)
    # 逻辑是这样实现的：先创建一个全0的矩阵，然后根据
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    # 对于无向图，邻接矩阵是对称的。上一步得到的adj是按有向图构建的，转换成无向图的邻接矩阵需要扩充成对称矩阵将i->j与j->i中权重最大的那个, 作为无向图的节点i与节点j的边权.
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    # 对应公式A~ = A+IN
    adj = normalize(adj + sp.eye(adj.shape[0]))  # eye创建单位矩阵，第一个参数为行数，第二个为列数

    # 分别构建训练集、验证集、测试集，并创建特征矩阵、标签向量和邻接矩阵的tensor，用来做模型的输入
    idx_train = range(140)  # 训练集
    idx_val = range(200, 500)   # 验证集
    idx_test = range(500, 1500)  # 测试集

    # todense()将矩阵显示出来，np.array创建数组，torch.FloatTensor（同torch.Tensor一样）创建单精度浮点型的张量
    features = torch.FloatTensor(np.array(features.todense()))  # tensor为pytorch常用的数据结构
    labels = torch.LongTensor(np.where(labels)[1])  # np.where()[0]表示行索引，np.where()[0]表示列索引
    adj = sparse_mx_to_torch_sparse_tensor(adj)  # tensor为pytorch常用的数据结构，将邻接矩阵转为tensor处理

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

# 矩阵归一化的实现
def normalize(mx):
    """Row-normalize sparse matrix"""
    # 论文里A^=(D~)^-1 A~这个公式
    rowsum = np.array(mx.sum(1))  # 对每一行求和

    # (D~)^-1
    r_inv = np.power(rowsum, -1).flatten()

    # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_inv[np.isinf(r_inv)] = 0.

    # 构建对角元素为r_inv的对角矩阵
    r_mat_inv = sp.diags(r_inv)

    # 论文里A^=(D~)^-1 A~这个公式
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    # 使用type_as(tesnor)将张量转换为给定类型的张量
    preds = output.max(1)[1].type_as(labels)
    # 记录等于preds的label eq:equal
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):  # 把一个sparse matrix转为torch稀疏张量
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    '''
        numpy中的ndarray转化成pytorch中的tensor : torch.from_numpy()
        pytorch中的tensor转化成numpy中的ndarray : numpy()
    '''
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
