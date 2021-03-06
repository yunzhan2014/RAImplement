# -*- coding: utf-8 -*-
"""
   module:进行数据集加载和交叉验证分集

   Aauthor： yunzhan

   Data: 6/8 2017

"""
import random
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise
from sklearn.model_selection import train_test_split


def cv_data(source_data, rate=0.2):
    """
    对数据集进行
    :param source_data: ratting信息的文件
    :param rate: 选择训练集与测试集的比例
    :return: 返回训练集与测试集
    """
    # header = ['userId', 'itemId', 'ratings','time']
    header = ['userId', 'itemId', 'ratings']
    data = pd.read_csv(source_data, sep=' ', header=None, names=header)
    user_num = data.userId.max()
    data.itemId = data.itemId + user_num
    train_data, test_data = train_test_split(data, test_size=rate)
    return train_data, test_data, user_num


def social_merge(rating_data, social_file):
    """
    merge social data on rating data
    :param rating_data:
    :param social_file:
    :return:
    """
    # header = ['user_s', 'user_t', 'trust_grade']
    header = ['userId', 'itemId', 'ratings']
    social_data = pd.read_csv(social_file, sep=' ', header=None, names=header)
    # 融合社交信息的几种方式
    np.random.seed(43)
    # rating_data.ratings = rating_data.ratings / 5
    # social_data.ratings = np.random.uniform(0, 1, social_data.shape[0])
    social_data.ratings = social_data.ratings * 5
    social_data.ratings = np.random.randint(1, 5, social_data.shape[0])
    frames = [rating_data, social_data]
    result = pd.concat(frames)
    print(result.head())
    print(result.tail())
    return result


def load_rating(path, main_keys="item", split_sig='\t'):
    # 获得item Id和 UserId 构建存储项目、用户、评分之间关系的数据结构
    setofItem = {}
    count = 0
    for line in open(path):
        (userId, itemId, rating) = line.split(split_sig)[0:3]
        if (main_keys == "item"):
            setofItem.setdefault(int(itemId), {})
            setofItem[int(itemId)][int(userId)] = float(rating)
            count = count + 1
        else:
            setofItem.setdefault(int(userId), {})
            setofItem[int(userId)][int(itemId)] = float(rating)
            count = count + 1

    print(len(setofItem))
    return setofItem


def loadTrust(path, split_sig='\t'):
    trustmatrix = {}  # 统计表中有多少个用户
    count = 0
    lastId = 0
    for line in open(path):
        (userId, trustUser, value) = line.split(split_sig)

        trustmatrix.setdefault(int(userId), {})
        trustmatrix[int(userId)][int(trustUser)] = float(value)

        if (userId != lastId):
            count = count + 1
        lastId = userId

    print(len(trustmatrix))
    return trustmatrix


def averageRating(user_item_dict):
    aver_rating_dict = {}
    for key, sec_dict in user_item_dict.items():
        aver_rating_dict[key] = 1.0 * sum(sec_dict.values()) / len(sec_dict)
    return aver_rating_dict


def delRedundatInfor(path, split_sig='\t', max_num=1508):
    '''
        去除trust文件里面的冗余信息
    '''
    trustmatrix = {}
    ###统计表中有多少个用户
    count = 0
    lastId = 0

    for line in open(path):
        (user, trusted_user, value) = line.split(split_sig)
        if (int(user) <= max_num and int(trusted_user) <= max_num):
            trustmatrix.setdefault(int(user), {})
            trustmatrix[int(user)][int(trusted_user)] = float(value)

            if (user != lastId):
                count = count + 1
            lastId = user

    print(len(trustmatrix))
    return trustmatrix


def transformPrefs(prefs):
    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})

            # Flip item and person
            result[item][person] = prefs[person][item]
    return result


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
