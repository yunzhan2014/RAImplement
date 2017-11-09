import pandas as pd
import numpy as np
import loadData as ld
from sklearn.neighbors import KDTree
from sklearn.metrics import pairwise


def similarity_matrix(file, user_numbers):
    """
    加载文件，同时计算出相似度矩阵
    :param file: 要加载文件的路进
    :param user_numbers
    :return:
    """
    with open(file) as f:
        table = pd.read_table(f, sep=' ', header=None, index_col=0, names=None, lineterminator='\n')
    table = table.sort_index(axis=0)
    label_index = np.asarray(table.index)
    table = (table - table.min()) / (table.max() - table.min())
    user_table = table[:user_numbers]
    item_table = table[user_numbers:]
    sim_matrix = pairwise.cosine_similarity(user_table, item_table)
    sim_matrix = np.asarray(sim_matrix)
    sim_matrix = pd.DataFrame(sim_matrix, index=label_index[:user_numbers], columns=label_index[user_numbers:])
    return sim_matrix


def large_graph_similarity_compute(file, user_num):
    with open(file) as f:
        table = pd.read_table(f, sep=' ', header=None, index_col=0, names=None, lineterminator='\n')
    table = table.sort_index(axis=0)
    table = (table - table.min()) / (table.max() - table.min())
    return table


def kd_tree_similarity(file, user_numbers, top_k_value, distance='euclidean'):
    """
    使用kd tree寻找最近邻"
    :param file:
    :param user_numbers:
    :param top_k_value:
    :param distance: 选择kd tree所采用的distance类型
    :return:
    """
    with open(file) as f:
        table = pd.read_table(f, sep=' ', header=None, index_col=0, names=None, lineterminator='\n')
    table = table.sort_index(axis=0)
    label_index = np.asarray(table.index)
    tree = KDTree(table, metric=distance, leaf_size=user_numbers)
    result = dict()
    for i in label_index:
        q_list = table.loc[i].reshape(1, -1)
        dist, ind = tree.query(q_list, k=top_k_value)
        result[i] = [label_index[j] for j in ind[0]]
    return result


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
        top_k_list.setdefault(key, list(basket.index))
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


def evaluation(top_k_dict, test_dict, k_val):
    """
    测试topK列表的推荐效果
    :param top_k_dict:
    :param test_dict:
    :param k_val:
    :return: precision, recall, ncrr
    """
    relevant_num = 0
    t_l = []
    recall = 0
    topk_dict_num = len(top_k_dict) * k_val
    ncrr = 0
    count = 0
    for key, value in top_k_dict.items():
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
    print('top_k_dict_num:' + str(topk_dict_num))
    print('NCRR:%.5f' % (ncrr / count))
    precision = relevant_num / topk_dict_num
    recall = recall / len(test_dict)
    return precision, recall


def main():
    header = ['userId', 'itemId', 'ratings']
    data_set_root = '/home/elics-lee/academicSpace/dataSet/FilmTrust'  # 运行的时输入本地路径

    train_data = pd.read_csv("%s/graph/train.csv" % data_set_root, sep=' ', names=header)
    test_data = pd.read_csv('%s/graph/test.csv' % data_set_root, sep=' ', names=header)  # 把训练集和测试集导入到内存当中
    top_k_num = 10
    user_num = train_data.userId.max()

    emb_file = '%s/emb/emb9.txt' % data_set_root

    # s, index_label = load_DataFrame2(emb_file)
    sim_matrix, label_index = load_DataFrame(emb_file)
    # user_matrix = similarity_matrix(emb_file, user_num)
    # item_matrix = user_matrix.T
    item_matrix, user_matrix = matrix_split(sim_matrix, user_num)

    user_dict_train, item_dict_train = construct_dict(train_data)
    user_dict_test, item_dict_test = construct_dict(test_data)  # 分别得到基于用户和基于物品的测试集合

    filter_user_matrix = filter_rating(user_matrix, user_dict_train)
    filter_item_matrix = filter_rating(item_matrix, item_dict_train)  # 分别对用户-物品和物品-用户的相似度矩阵剔除train data当中集合

    user_top_list = top_k_list(filter_user_matrix, top_k_num)
    item_top_list = top_k_list(filter_item_matrix, top_k_num)  # 分别得到基于用户和项目的物品的topK推荐列表

    user_precision, user_recall = evaluation(user_top_list, user_dict_test, top_k_num)  # 分别得到基于用户和项目的物品的topK推荐列表的评价指标
    item_precision, item_recall = evaluation(item_top_list, item_dict_test, top_k_num)

    print('User-based embedding Precision: %.5f' % user_precision)  # 打印输出结果
    print('User-based embedding recall: %.5f' % user_recall)
    print('Item-based embedding Precision: %.5f' % item_precision)
    print('Item-based embedding recall: %.6f' % item_recall)


def neighbor():
    header = ['userId', 'itemId', 'ratings']
    data_set_root = '/home/elics-lee/academicSpace/dataSet/FilmTrust'
    # 把训练集和测试集导入到内存当中
    train_data = pd.read_csv("%s/graph/train.csv" % data_set_root, sep=' ', names=header)
    test_data = pd.read_csv('%s/graph/test.csv' % data_set_root, sep=' ', names=header)
    top_k_value = 10
    user_num = train_data.userId.max()
    emb_file = '%s/emb/emb9.txt' % data_set_root
    # 分别得到基于用户和基于物品的测试集合
    user_dict_train, item_dict_train = construct_dict(train_data)
    user_dict_test, item_dict_test = construct_dict(test_data)

    top_list = kd_tree_similarity(emb_file, user_num, top_k_value)
    keys = list(top_list.keys())

    user_keys = [i for i in keys if i <= user_num]
    item_keys = [i for i in keys if i > user_num]
    user_top_list = {k: top_list[k] for k in user_keys}
    item_top_list = {k: top_list[k] for k in item_keys}

    user_top_list = filter_kd_tree(user_top_list, user_dict_train)
    item_top_list = filter_kd_tree(item_top_list, item_dict_train)
    # 分别得到基于用户和项目的物品的topK推荐列表的评价指标
    user_precision, user_recall = evaluation(user_top_list, user_dict_test, top_k_value)
    item_precision, item_recall = evaluation(item_top_list, item_dict_test, top_k_value)

    print('User-based embedding Precision: %.5f' % user_precision)
    print('User-based embedding recall: %.5f' % user_recall)
    print('Item-based embedding Precision: %.5f' % item_precision)
    print('Item-based embedding recall: %.6f' % item_recall)


if __name__ == "__main__":
    main()
