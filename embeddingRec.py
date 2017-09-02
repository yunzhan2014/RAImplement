# -*- coding: utf-8 -*-
from os import path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise
from sklearn.neighbors import KDTree

import loadData as lD
from trainNode2vec import Node2vec


def base_cosine_similarity_list(node2vec_entity, top_k_value):
    """
    根据余弦相似性得到top list
    :param node2vec_entity:
    :param top_k_value
    :return:
    """

    data_set_path = node2vec_entity.data_set_path
    user_num = node2vec_entity.user_num
    train_data = node2vec_entity.train_data
    test_data = node2vec_entity.test_data
    emb_file = node2vec_entity.emb_file

    top_k_value = top_k_value
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
    user_result = evaluation(user_top_list, user_dict_test)
    item_result = evaluation(item_top_list, item_dict_test)
    # 把评估的结果写入到result文件当中
    result_file = "%s/result/result.csv" % data_set_path
    result_list = list()
    result_list.append("cosine")
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


def base_kd_tree_similarity_list(node2vec_entity, top_k_value, is_social=True):
    """
    :param node2vec_entity:
    :param top_k_value
    :param is_social
    :return:
    """
    # 初始化一些参数
    header = ['user', 'item', 'rating']
    data_set_path = node2vec_entity.source_data_path
    train_data = node2vec_entity.train_data
    test_data = node2vec_entity.test_data
    print(train_data, test_data)
    train_data = pd.read_csv(train_data, sep=' ', names=header)
    test_data = pd.read_csv(test_data, sep=' ', names=header)
    emb_file = node2vec_entity.output_file
    user_num = node2vec_entity.user_num
    top_k_value = top_k_value
    # print("这一组的top k value 是：%d" % top_k_value)
    # 分别得到基于用户和基于物品的训练集合与测试集合
    user_dict_train, item_dict_train = construct_dict(train_data)
    user_dict_test, item_dict_test = construct_dict(test_data)
    # 根据最近生成top list
    user_top_list, item_top_list = kd_tree_similarity(emb_file, user_num)
    # 把top list分割成user:item list和item:user list的形式
    # keys = list(top_list.keys())
    # user_keys = [i for i in keys if i <= user_num]
    # item_keys = [i for i in keys if i > user_num]
    # user_top_list = {k: top_list[k] for k in user_keys}
    # item_top_list = {k: top_list[k] for k in item_keys}
    # 过滤掉train data中已经包含的评分信息(目前会导致top list变小)
    user_top_list = filter_kd_tree(user_top_list, user_dict_train, top_k_value)
    item_top_list = filter_kd_tree(item_top_list, item_dict_train, top_k_value)
    # 分别得到基于用户和项目的物品的topK推荐列表的评价指标
    user_result = evaluation(user_top_list, user_dict_test)
    item_result = evaluation(item_top_list, item_dict_test)
    # 把评估的结果写入到result文件当中
    result_file = "%s/result/result.csv" % data_set_path
    result_list = list()
    if is_social:
        result_list.append("social_merge")
    else:
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


def kd_tree_similarity(file, user_numbers, distance='euclidean'):
    """
    使用kd tree寻找最近邻"
    :param file:
    :param user_numbers: 作为kd tree的leaf_size
    :param distance: 选择kd tree所采用的distance类型
    :return: dict: id号对应的一系列id
    """
    with open(file) as f:
        node_num, dimension = f.readline().split()
        table = pd.read_table(f, sep=' ', header=None, index_col=0, names=None, lineterminator='\n')
    print("Embedding node number is %s, dimension is %s" % (node_num, dimension))
    table = table.sort_index(axis=0)
    # normalize row
    table = table.div(np.sqrt(table.pow(2).sum()), axis=0)
    table = table.fillna(0)
    # split table to user table and item table
    label_index = np.asarray(table.index)
    user_label_index = label_index[label_index <= user_numbers]
    item_label_index = label_index[label_index > user_numbers]
    # 根据user label index筛选出user table
    user_table = table.ix[user_label_index]
    item_table = table.ix[item_label_index]
    # 基于用户挑选的最近邻列表
    tree = KDTree(item_table, metric=distance, leaf_size=user_numbers)
    user_result = dict()
    for i in user_table.iterrows():
        user_i = list()
        user_i.append(i[1].tolist())
        dist, ind = tree.query(user_i, k=100)
        user_result[i[0]] = [item_label_index[j] for j in ind[0]]
    # 基于物品挑选的最近邻列表
    tree = KDTree(user_table, metric=distance, leaf_size=user_numbers)
    item_result = dict()
    for i in item_table.iterrows():
        item_i = list()
        item_i.append(i[1].tolist())
        dist, ind = tree.query(item_i, k=100)
        item_result[i[0]] = [user_label_index[j] for j in ind[0]]
    return user_result, item_result


def filter_kd_tree(top_dict, train_dict, top_k_value):
    """
    对采用kd tree方式的top list进行训练集的筛选和过滤
    :param top_dict:
    :param train_dict:
    :param top_k_value
    :return: dict
    """
    result = dict()
    for key, value in train_dict.items():
        if key in top_dict:
            row_list = top_dict[key]
            list_intersection = set(value).intersection(row_list)
            l = [x for x in row_list if x not in list_intersection]
            result[key] = l[0:top_k_value]
            # print("top %d in user or item %d" % (len(value), key))
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


def evaluation(top_dict, test_dict):
    """
    测试topK列表的推荐效果
    :param top_dict:
    :param test_dict:
    :return: precision, recall, ncrr
    """
    relevant_num = 0
    t_l = []
    recall = 0
    top_dict_num = 0
    ncrr = 0
    count = 0
    result = dict()
    for key, value in top_dict.items():
        top_dict_num += len(value)
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
    is_social = False
    # is_social = True
    # 训练模型参数
    # d = 10
    # max_iter = 1
    # walk_length = 80
    # n_walks = 80
    # window_size = 80
    # ret_p = 1
    # inout_p = 2
    model_parameter = ['10', '1', '80', '80', '80', '1', '1']

    # 选定数据集位置,对数据集进行分集
    source_data_path = "/home/elics-lee/academicSpace/dataSet/epinions"
    ratings_file = "%s/ratings.txt" % source_data_path
    train_data, test_data, user_num = lD.cv_data(ratings_file, rate=0.2)

    if is_social:
        social_file = "%s/trust.txt" % source_data_path
        train_data = lD.social_merge(train_data, social_file, user_num)

    train_data_file = "%s/train.csv" % source_data_path
    test_data_file = "%s/test.csv" % source_data_path
    train_data.to_csv(train_data_file, sep=" ", index=None, header=None, columns=None)
    test_data.to_csv(test_data_file, sep=" ", index=None, header=None, columns=None)

    # 模型训练
    model_parameter = [source_data_path] + model_parameter
    if is_social:
        emb_file_name = "%s/emb/social_dim%s_iter%s_walklen%s_walkper%s_window%s_p%s_q%s.emb" % tuple(model_parameter)
    else:
        emb_file_name = "%s/emb/dim%s_iter%s_walklen%s_walkper%s_window%s_p%s_q%s.emb" % tuple(model_parameter)
    model_parameter = model_parameter + [emb_file_name]
    print(model_parameter)
    node2vec = Node2vec(user_num, model_parameter)
    if not path.exists(emb_file_name):
        node2vec.train_model()
    # 模型检验
    base_kd_tree_similarity_list(node2vec, 10, is_social)

if __name__ == "__main__":
    main()
