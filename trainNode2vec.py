import loadData as lD
from subprocess import call


class Node2vec(object):
    """
    通过node2vec方法生成的低维向量的实体
    包含train_data,test_data
    embedding_file等一系列的属性信息
    """
    header = ['userId', 'itemId', 'ratings']
    data_set_path = '/home/elics-lee/academicSpace/dataSet/ciao'

    def __init__(self, input_file, d, max_iter, walk_length, n_walks, window_size, ret_p, inout_p):
        """
        :param self:
        :param embedding file:
        :return:
        """
        self._input_file = input_file
        self._d = d
        self._max_iter = max_iter
        self._walkLength = walk_length
        self._numWalks = n_walks
        self._contextSize = window_size
        self._return_p = ret_p
        self._inout_p = inout_p
        self._method_name = 'node2vec'

    def train_model(self):
        args = ["./node2vec", "-i:%s" % self._input_file, "-o:tempGraph.emb", "-d:%d" % self._d,
                "-l:%d" % self._walkLength, "-r:%d" % self._numWalks, "-k:%d" % self._contextSize,
                "-e:%d" % self._max_iter, "-p:%f" % self._return_p, "-q:%f" % self._inout_p, "-v", "-dr", "-w"]
        call(args)

# 选定数据集位置
source_data_path = "/home/elics-lee/academicSpace/dataSet/ciao"
ratings_file = "%s/ratings.txt" % source_data_path
social_file = "%s/trust.txt" % source_data_path
# 对数据集进行分集
train_data, test_data = lD.cv_data(ratings_file, rate=0.2)
train_data_file = "%s/train.csv" % source_data_path
test_data_file = "%s/test.csv" % source_data_path
train_data.to_csv(train_data_file, sep=" ", index=None, header=None, columns=None)
test_data.to_csv(test_data_file, sep=" ", index=None, header=None, columns=None)
# 训练模型
d = 10
input_file = train_data_file
max_iter = 1
walk_length = 80
n_walks = 80
window_size = 80
ret_p = 1
inout_p = 2
node2vec = Node2vec(input_file, d, max_iter, walk_length, n_walks, window_size, ret_p, inout_p)
node2vec.train_model()

# def social_add(train_data, social_data):
