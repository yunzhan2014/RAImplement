import pandas as pd
import numpy as np
from sklearn.metrics import pairwise
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine


def cv_data(source_data, rate=0.2):
    """
    对数据集进行分集
    :param source_data:
    :param rate:
    :return:
    """
    # header = ['userId', 'itemId', 'ratings','time']
    header = ['userId', 'itemId', 'ratings']

    data = pd.read_csv(source_data, sep=' ', header=None, names=header)
    user_num = data.userId.unique().shape[0]
    data.itemId = data.itemId + user_num
    data.ratings = data.ratings / 5
    train_data, test_data = train_test_split(data, test_size=rate)
    return train_data, test_data


def load_DataFrame(file):
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


def similarity_matrix(file, user_numbers):
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
    user_table = table[:user_numbers]
    item_table = table[user_numbers:]
    sim_matrix = pairwise.cosine_similarity(user_table, item_table)
    sim_matrix = np.asarray(sim_matrix)
    sim_matrix = pd.DataFrame(sim_matrix, index=label_index[:user_numbers], columns=label_index[user_numbers:])
    return sim_matrix


def load_DataFrame2(file):
    """
    针对节点数较多的矩阵所导致的相似度矩阵计算困难的情况进行处理
    :param file:
    :return:
    """
    count = 0
    with open(file) as f:
        table = pd.read_table(f, sep=' ', header=None, index_col=0, names=None, lineterminator='\n')
    table = table.sort_index(axis=0)
    label_index = np.asarray(table.index)
    # table = table.abs()
    table = (table - table.min()) / (table.max() - table.min())
    # table = (table - table.mean()) / table.std()
    s = np.asarray(table)
    pniece_s = np.array_split(s, 4)
    full_sim_narray = np.array([], dtype=None).reshape(0, len(s))
    for i in pniece_s:
        print(count)
        row_sim_narray = np.array([], dtype=float).reshape(len(pniece_s[count]), 0)
        for j in pniece_s:
            print("row_array initialize shape is {0}".format(np.shape(row_sim_narray)))
            pniece_sim_narray = pairwise.cosine_similarity(i, j)
            print("sim_array shape is {0}".format(np.shape(pniece_sim_narray)))
            row_sim_narray = np.hstack((pniece_sim_narray, row_sim_narray))
            del pniece_sim_narray
            print("row_sim_array shape is {0}".format(np.shape(row_sim_narray)))
        print("full_array initialize shape is {0}".format(np.shape(full_sim_narray)))
        full_sim_narray = np.vstack((full_sim_narray, row_sim_narray))
        del row_sim_narray
        count += 1

    return pniece_s, full_sim_narray


def large_graph_simlarity_compute(file, user_num):
    with open(file) as f:
        table = pd.read_table(f, sep=' ', header=None, index_col=0, names=None, lineterminator='\n')
    table = table.sort_index(axis=0)
    label_index = np.asarray(table.index)
    # table = table.abs()
    table = (table - table.min()) / (table.max() - table.min())
    # table = (table - table.mean()) / table.std()
    user_vector = table[0]
    return table


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
    top_k_list = dict()
    for key, row in pd_matrix_like.iterrows():
        basket = row.sort_values()[::-1][0:k]
        top_k_list.setdefault(key, list(basket.index))
    return top_k_list


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
        filter_matrix.loc[[key], value] = 0
    return filter_matrix


def evaluation(topk_dict, test_dict, k_val):
    """
    测试topK列表的推荐效果
    :param topk_dict:
    :param test_dict:
    :param k_val:
    :return: precision, recall, ncrr
    """
    relevant_num = 0
    t_l = []
    recall = 0
    topk_dict_num = len(topk_dict) * k_val
    ncrr = 0
    count = 0
    for key, value in topk_dict.items():
        # rank = 0
        if key in test_dict:
            overlap_set = set(value).intersection(test_dict[key])
            row_overlap_num = len(overlap_set)
            if row_overlap_num != 0:
                count += 1
                relevant_num += row_overlap_num
                rank_index = [1 / (1 + value.index(s)) for s in overlap_set]
                idel_crr = sum(1 / np.arange(1, row_overlap_num + 1))
                ncrr += sum(rank_index) / idel_crr
                t_l.append(row_overlap_num)
            recall += row_overlap_num / len(test_dict[key])

    print('relevant num:' + str(relevant_num))
    print('topk_dict_num:' + str(topk_dict_num))
    print('ncrr:%.5f' % (ncrr / count))
    precision = relevant_num / topk_dict_num
    recall = recall / len(test_dict)
    return precision, recall


def main():
    header = ['userId', 'itemId', 'ratings']  # 三元组的属性名称

    dataset_root = '/home/elics-lee/acdamicSpace/dataset/ciao'  # 运行的时输入本地路径

    train_data = pd.read_csv("%s/graph/train.csv" % dataset_root, sep=' ', names=header)
    test_data = pd.read_csv('%s/graph/test.csv' % dataset_root, sep=' ', names=header)  # 把训练集和测试集导入到内存当中
    topk_value = 10
    user_num = train_data.userId.max()

    emb_file = '%s/emb/emb.txt' % dataset_root

    # s, index_label = load_DataFrame2(emb_file)
    # sim_matrix, label_index = load_DataFrame(emb_file)
    user_matrix = similarity_matrix(emb_file, user_num)
    item_matrix = user_matrix.T
    user_dict_train, item_dict_train = construct_dict(train_data)
    user_dict_test, item_dict_test = construct_dict(test_data)  # 分别得到基于用户和基于物品的测试集合

    filter_user_matrix = filter_rating(user_matrix, user_dict_train)
    filter_item_matrix = filter_rating(item_matrix, item_dict_train)  # 分别对用户-物品和物品-用户的相似度矩阵剔除train data当中集合

    user_top_list = top_k_list(filter_user_matrix, topk_value)
    item_top_list = top_k_list(filter_item_matrix, topk_value)  # 分别得到基于用户和项目的物品的topK推荐列表

    user_precision, user_recall = evaluation(user_top_list, user_dict_test, topk_value)  # 分别得到基于用户和项目的物品的topK推荐列表的评价指标
    item_precision, item_recall = evaluation(item_top_list, item_dict_test, topk_value)

    print('User-based embeding Precision: %.5f' % (user_precision))  # 打印输出结果
    print('User-based embeding recall: %.5f' % (user_recall))
    print('Item-based embeding Precision: %.5f' % (item_precision))
    print('Item-based embeding recall: %.6f' % (item_recall))

if __name__ == "__main__":
    main()