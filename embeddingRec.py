import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.metrics import pairwise


class Node2vec(object):
    """
    通过node2vec方法生成的低维向量的实体
    包含train_data,test_data
    embedding_file等一系列的属性信息
    """
    header = ['userId', 'itemId', 'ratings']
    data_set_path = '/home/elics-lee/academicSpace/dataSet/ciao'

    def __init__(self, embedding_file, top_k_value):
        """
        :param self:
        :param embedding_file:
        :return:
        """

        # 把训练集和测试集导入到内存当中
        self.train_data = pd.read_csv("%s/graph/train.csv" % self.data_set_path, sep=' ', names=self.header)
        self.test_data = pd.read_csv('%s/graph/test.csv' % self.data_set_path, sep=' ', names=self.header)
        self.top_k_value = top_k_value
        self.user_num = self.train_data.userId.max()
        self.emb_file = '%s/emb/%s' % (self.data_set_path, embedding_file)


def base_cosine_similarity_list(embed_entity):
    """
    根据余弦相似性得到top list
    :param embed_entity:
    :return:
    """

    data_set_path = embed_entity.data_set_path
    user_num = embed_entity.user_num
    train_data = embed_entity.train_data
    test_data = embed_entity.test_data
    top_k_value = embed_entity.top_k_value
    emb_file_list = ["embl_1.txt", "embl_2.txt", "embl_3.txt", "embl_4.txt", "embl_5.txt", "embl_6.txt", "embl_7.txt",
                     "embl_8.txt", "embl_9.txt"]
    for i in emb_file_list:
        emb_file = '%s/emb/' % data_set_path
        emb_file = emb_file + str(i)
        print(emb_file)
        sim_matrix, label_index = cosine_similarity(emb_file)
        item_matrix, user_matrix = matrix_split(sim_matrix, user_num)
        # 分别得到基于用户和基于物品的测试集合
        user_dict_train, item_dict_train = construct_dict(train_data)
        user_dict_test, item_dict_test = construct_dict(test_data)
        # 分别对用户-物品和物品-用户的相似度矩阵剔除train data当中集合
        filter_user_matrix = filter_rating(user_matrix, user_dict_train)
        filter_item_matrix = filter_rating(item_matrix, item_dict_train)
        # 分别得到基于用户和项目的物品的topK推荐列表
        user_top_list = top_k_list(filter_user_matrix, top_k_value)
        item_top_list = top_k_list(filter_item_matrix, top_k_value)
        # 分别得到基于用户和项目的物品的topK推荐列表的评价指标
        user_result = evaluation(user_top_list, user_dict_test, top_k_value)
        item_result = evaluation(item_top_list, item_dict_test, top_k_value)
        # 把评估的结果写入到result文件当中
        result_file = "%s/result/result.csv" % data_set_path
        result_list = list()
        result_list.append(i)
        result_list.append(str(top_k_value))
        result_list.append("{:06.5f}".format(user_result['precision']))
        result_list.append("{:06.5f}".format(user_result['recall']))
        result_list.append("{:06.5f}".format(user_result['ncrr']))
        result_list.append("{:06.5f}".format(item_result['precision']))
        result_list.append("{:06.5f}".format(item_result['recall']))
        result_list.append("{:06.5f}".format(item_result['ncrr']))
        with open(result_file, 'a') as f:
            f.write(','.join(result_list) + "\n")
        f.close()


def base_kd_tree_similarity_list(embed_entity):
    """
    对
    :param embed_entity:
    :return:
    """
    # 初始化一些参数
    data_set_path = embed_entity.data_set_path
    train_data = embed_entity.train_data
    test_data = embed_entity.test_data
    emb_file = embed_entity.emb_file
    user_num = embed_entity.user_num
    top_k_value = embed_entity.top_k_value
    # 分别得到基于用户和基于物品的训练集合与测试集合
    user_dict_train, item_dict_train = construct_dict(train_data)
    user_dict_test, item_dict_test = construct_dict(test_data)
    # 根据最近生成top list
    top_list = kd_tree_similarity(emb_file, user_num, top_k_value)
    # 把top list分割成user:item list和item:user list的形式
    keys = list(top_list.keys())
    user_keys = [i for i in keys if i <= user_num]
    item_keys = [i for i in keys if i > user_num]
    user_top_list = {k: top_list[k] for k in user_keys}
    item_top_list = {k: top_list[k] for k in item_keys}
    # 过滤掉train data中已经包含的评分信息(目前会导致top list变小)
    user_top_list = filter_kd_tree(user_top_list, user_dict_train)
    item_top_list = filter_kd_tree(item_top_list, item_dict_train)
    # 分别得到基于用户和项目的物品的topK推荐列表的评价指标
    user_result = evaluation(user_top_list, user_dict_test, top_k_value)
    item_result = evaluation(item_top_list, item_dict_test, top_k_value)
    # 把评估的结果写入到result文件当中
    result_file = "%s/result/result.csv" % data_set_path
    result_list = list()
    result_list.append("kd_tree")
    result_list.append(str(top_k_value))
    result_list.append("{:06.5f}".format(user_result['precision']))
    result_list.append("{:06.5f}".format(user_result['recall']))
    result_list.append("{:06.5f}".format(user_result['ncrr']))
    result_list.append("{:06.5f}".format(item_result['precision']))
    result_list.append("{:06.5f}".format(item_result['recall']))
    result_list.append("{:06.5f}".format(item_result['ncrr']))
    with open(result_file, 'a') as f:
        f.write(','.join(result_list) + "\n")
    f.close()


def kd_tree_similarity(file, user_numbers, top_k_value, distance='euclidean'):
    """
    使用kd tree寻找最近邻"
    :param file:
    :param user_numbers: 作为kd tree的leaf_size
    :param top_k_value:
    :param distance: 选择kd tree所采用的distance类型
    :return:
    """
    with open(file) as f:
        table = pd.read_table(f, sep=' ', header=None, index_col=0, names=None, lineterminator='\n')
    table = table.sort_index(axis=0)
    table = table.div(np.sqrt(table.pow(2).sum()), axis=0)
    table = table.fillna(0)
    label_index = np.asarray(table.index)
    tree = KDTree(table, metric=distance, leaf_size=user_numbers)
    result = dict()
    for i in label_index:
        # q_list = table.loc[i].reshape(1, -1)
        q_list = np.reshape(table.loc[i], (1, -1))
        dist, ind = tree.query(q_list, k=top_k_value * 35)
        result[i] = [label_index[j] for j in ind[0]]
    return result


def cosine(x, y):
    return pairwise.cosine_distances(x, y)


def filter_kd_tree(neighbor_list, data_dict):
    """
    对采用kd tree方式的top list进行训练集的筛选和过滤
    :param neighbor_list:
    :param data_dict:
    :return:
    """
    result = dict()
    for key, value in data_dict.items():
        if key in neighbor_list:
            row_list = neighbor_list[key]
            list_intersection = set(value).intersection(row_list)
            l = [x for x in value if x not in list_intersection]
            result[key] = l
    return result


def cosine_similarity(file):
    """
    加载文件，同时计算出相似度矩阵
    :param file: 要加载文件的路进
    :return:
    """
    with open(file) as f:
        table = pd.read_table(f, sep=' ', header=None, index_col=0, names=None, lineterminator='\n')
    table = table.sort_index(axis=0)
    label_index = np.asarray(table.index)
    # table = table.abs()
    table = (table - table.min()) / (table.max() - table.min())
    # table = (table - table.mean()) / table.std()
    s = np.asarray(table)
    sim_matrix = pairwise.cosine_similarity(s)
    sim_matrix = pd.DataFrame(sim_matrix, index=label_index, columns=label_index)
    return sim_matrix, label_index


def matrix_split(sim_matrix, user_num):
    """
    对给的相似度按照物品-用户相似度，用户-物品相似度进行划分
    :param sim_matrix:
    :param user_num:
    :return: 物品相似度矩阵，用户-物品相似度矩阵
    """
    item_matrix = sim_matrix.loc[user_num + 1:, 0:user_num]
    user_matrix = sim_matrix.loc[0:user_num, user_num + 1:]
    return item_matrix, user_matrix


def top_k_list(pd_matrix_like, k):
    """
    根据经过剔除了train data里面的评分信息的矩阵得到针对每个用户TopK列表
    :param pd_matrix_like:
    :param k:
    :return:
    """
    result = dict()
    for key, row in pd_matrix_like.iterrows():
        basket = row.sort_values()[::-1][0:k]
        result.setdefault(key, list(basket.index))
    return result


def construct_dict(data):
    """
    对测试集，进行梳理，每个用户或物品形成一个Map,key为物品或者用户的ID,value为评价的集合
    :param data:
    :return:
    """
    user_dict = {}
    item_dict = {}
    for row in data.itertuples():
        user_dict.setdefault(row[1], []).append(row[2])
        item_dict.setdefault(row[2], []).append(row[1])
    return user_dict, item_dict


def filter_rating(data_matrix, data_dict):
    """
    过滤掉训练集里面已有的关联的评分(这里最好设置一个阈值,需要么？？)
    :param data_matrix:
    :param data_dict:
    :return:
    """
    filter_matrix = data_matrix.copy()
    for key, value in data_dict.items():
        if key in filter_matrix.index:
            row_list = list(filter_matrix.loc[key])
            list_intersection = set(value).intersection(row_list)
            filter_matrix.loc[[key], list_intersection] = 0
    return filter_matrix


def evaluation(top_dict, test_dict, k_val):
    """
    测试topK列表的推荐效果
    :param top_dict:
    :param test_dict:
    :param k_val:
    :return: precision, recall, ncrr
    """
    relevant_num = 0
    t_l = []
    recall = 0
    top_dict_num = len(top_dict) * k_val
    ncrr = 0
    count = 0
    result = dict()
    for key, value in top_dict.items():
        # rank = 0
        if key in test_dict:
            overlap_set = set(value).intersection(test_dict[key])
            row_overlap_num = len(overlap_set)
            if row_overlap_num != 0:
                count += 1
                relevant_num += row_overlap_num
                rank_index = [1 / (1 + value.index(s)) for s in overlap_set]
                ideal_crr = sum(1 / np.arange(1, row_overlap_num + 1))
                ncrr += sum(rank_index) / ideal_crr
                t_l.append(row_overlap_num)
            recall += row_overlap_num / len(test_dict[key])

    print('relevant num:' + str(relevant_num))
    print('top_k_dict_num:' + str(top_dict_num))
    result['precision'] = relevant_num / top_dict_num
    result['recall'] = recall / len(test_dict)
    result['ncrr'] = ncrr / count
    return result


def main():
    node2vec = Node2vec("embl_1.txt", 40)
    # base_cosine_similarity_list(node2vec)
    base_kd_tree_similarity_list(node2vec)


if __name__ == "__main__":
    main()
