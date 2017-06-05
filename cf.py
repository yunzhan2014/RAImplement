import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt

topk_value = 10
np.seterr(divide='ignore', invalid='ignore')
header = ['user_id', 'item_id', 'rating']
# df = pd.read_csv('/home/elics-lee/acdamicSpace/dataset/ml-100k/u.data', sep='\t', names=header)
df = pd.read_csv('/home/elics-lee/opt/node2vec/graph/FilmTrust/ratings.txt', sep=' ', names=header)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

train_data, test_data = train_test_split(df, test_size=0.2)

# Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

user_similarity = cosine_similarity(train_data_matrix)
item_similarity = cosine_similarity(train_data_matrix.T)

x, y = train_data_matrix.nonzero()


def rating_predict(ratings, similarity, type='user'):
    temp = ratings.copy()
    for i, j in zip(x, y):
        temp[i][j] = 1

    if type == 'user':
        pred = similarity.dot(ratings) / similarity.dot(temp)
        pred = np.nan_to_num(pred)
    elif type == 'item':
        pred = ratings.dot(similarity) / temp.dot(similarity)
        pred = np.nan_to_num(pred)

    for i, j in zip(x, y):
        pred[i][j] = 0
    return pred


user_prediction = rating_predict(train_data_matrix, user_similarity, type='user')
item_prediction = rating_predict(train_data_matrix, item_similarity, type='item')


def top_k_list(prediction_matrix, k_value):
    index_num = 1
    topk_list = dict()
    for row in prediction_matrix:
        basket = row.argsort()[::-1][0:k_value]
        topk_list.setdefault(index_num, list(basket))
        index_num += 1
    return topk_list


def construct_dict(data):
    user_dict = {}
    item_dict = {}
    for row in data.itertuples():
        user_dict.setdefault(row[1], []).append(row[2])
        item_dict.setdefault(row[2], []).append(row[1])
    return user_dict, item_dict


def filter_rating(data_matrix, data_dict):
    filter_matrix = data_matrix.copy()
    for key, value in data_dict.items():
        filter_matrix.loc[key, value] = 0
    return filter_matrix


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


def evaluation(topk_dict, test_dict, k_val, test_dict_num):
    relevant_num = 0
    topk_dict_num = len(topk_dict) * k_val

    for key, value in topk_dict.items():
        # rank = 0
        if key in test_dict:
            overlap_set = set(value).intersection(test_dict[key])
            row_overlap_num = len(overlap_set)
            relevant_num += row_overlap_num
            # for i in overlap_set:
            #    rank += ((i/value.index(i)) for i in overlap_set)

    print('relevant num:' + str(relevant_num))
    print('topk_dict_num:' + str(topk_dict_num))
    print('test_dict_num:' + str(test_dict_num))

    precision = relevant_num / topk_dict_num
    recall = relevant_num / test_dict_num
    return precision, recall


print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

item_top_list = top_k_list(item_prediction, topk_value)
user_top_list = top_k_list(user_prediction, topk_value)
user_dict_test, item_dict_test = construct_dict(test_data)
user_precision, user_recall = evaluation(item_top_list, user_dict_test, topk_value, len(test_data))
item_precision, item_recall = evaluation(item_top_list, item_dict_test, topk_value, len(test_data))

print('User-based CF Precision: %.5f' %(user_precision))
print('User-based CF recall: %.5f' %(user_recall))
print('Item-based CF Precision: %.5f' %(item_precision))
print('Item-based CF recall: %.5f' %(item_recall))
# import scipy.sparse as sp
# from scipy.sparse.linalg import svds
#
# get SVD components from train matrix. Choose k.
# u, s, vt = svds(train_data_matrix, k=20)
# s_diag_matrix = np.diag(s)
# X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
# print('User-based CF MSE: ' + str(rmse(X_pred, test_data_matrix)))
