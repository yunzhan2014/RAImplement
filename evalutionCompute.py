"""
    Function: 计算推荐系统评估参数

    Author: Elics Lee(yunzhan)

    Date: 12/21 2016

"""
import numpy as np
import math
from sklearn.metrics import mean_squared_error


def compute_rmse(peridict_ri, ri):
    length = len(peridict_ri)
    num_rui = 0
    sum_r = 0.0
    for i in range(length):
        if (ri[i] != 0):
            if (peridict_ri[i] != 0):
                print(peridict_ri[i])
            num_rui = num_rui + 1
            sum_r = sum_r + math.pow((peridict_ri[i] - ri[i]), 2)
    print(num_rui)
    return math.sqrt(sum_r / num_rui)


def compute_mae(perdict_ri, ri):
    length = len(perdict_ri)
    common_rating_num = 0
    sum_r = 0.0
    for i in range(length):
        if (ri[i] != 0):
            common_rating_num += 1
            sum_r += np.absolute(perdict_ri[i] - ri[i])
    return sum_r / common_rating_num


def rmse(prediction, ground_truth):
    """
    data structure is numpy array
    """
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return math.sqrt(mean_squared_error(prediction, ground_truth))


def evaluation(topk_dict, test_dict, k_val, test_dict_num):
    relevant_num = 0
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

    print('relevant num:' + str(relevant_num))
    print('topk_dict_num:' + str(topk_dict_num))
    print('test_dict_num:' + str(test_dict_num))
    print('ncrr:%s' % (ncrr / count))
    precision = relevant_num / topk_dict_num
    recall = relevant_num / test_dict_num
    return precision, recall
