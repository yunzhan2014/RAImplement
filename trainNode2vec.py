# -*- coding: utf-8 -*-
from subprocess import call


class Node2vec(object):
    """
    通过node2vec方法生成的低维向量的实体
    包含train_data,test_data
    embedding_file等一系列的属性信息
    """

    def __init__(self, user_numbers, parameter):
        """
        :param self:
        :param embedding file:
        :param user_numbers
        :param parameter
        :return:
        """
        # model parameter
        self._input_file = "%s/train.csv" % parameter[0]
        self._d = int(parameter[1])
        self._max_iter = int(parameter[2])
        self._walkLength = int(parameter[3])
        self._numWalks = int(parameter[4])
        self._contextSize = int(parameter[5])
        self._return_p = int(parameter[6])
        self._inout_p = int(parameter[7])
        self._method_name = 'node2vec'

        # ratings file information
        self.source_data_path = parameter[0]
        self.train_data = "%s/train.csv" % self.source_data_path
        self.test_data = "%s/test.csv" % self.source_data_path
        self.user_num = user_numbers
        self.output_file = parameter[8]

    def train_model(self):
        args = ["./node2vec", "-i:%s" % self._input_file, "-o:%s" % self.output_file, "-d:%d" % self._d,
                "-l:%d" % self._walkLength, "-r:%d" % self._numWalks, "-k:%d" % self._contextSize,
                "-e:%d" % self._max_iter, "-p:%f" % self._return_p, "-q:%f" % self._inout_p, "-v", "-dr", "-w"]
        call(args)
