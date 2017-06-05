import pandas as pd
import numpy as np
from sklearn.metrics import pairwise
from sklearn.model_selection import train_test_split

#dataset_file_path = '/home/elics-lee/acdamicSpace/dataset/ml-1m/ratings.dat'
dataset_file_path = 'graph/FilmTrust/ratings.txt'
def cv_data(source_data, rate=0.2):
    # header = ['userId', 'itemId', 'ratings','time']
    header = ['userId', 'itemId', 'ratings']
    data = pd.read_csv(source_data, sep=' ', header=None, names=header)
    user_num = data.userId.unique().shape[0]
    data.itemId = data.itemId + user_num
    train_data, test_data = train_test_split(data, test_size=rate)
    return train_data, test_data

def load_DataFrame(file):
    with open(file) as f:
        table = pd.read_table(f, sep=' ', header=None, index_col=0, names=None, lineterminator='\n')
    table = table.sort_index(axis=0)
    label_index = np.asarray(table.index)
    #table = table.abs()
    table = (table - table.min()) / (table.max() - table.min())
    #table = (table - table.mean()) / table.std()
    s = np.asarray(table)
    sim_matrix = pairwise.cosine_similarity(s)
    sim_matrix = pd.DataFrame(sim_matrix, index=label_index, columns=label_index)
    return sim_matrix, label_index


def matrix_split(sim_matrix, user_num):
    item_matrix = sim_matrix.loc[user_num + 1:, 0:user_num]
    user_matrix = sim_matrix.loc[0:user_num, user_num + 1:]
    return item_matrix, user_matrix


def top_k_list(pd_matrix_like, k):
    top_k_list = dict()
    for key, row in pd_matrix_like.iterrows():
        basket = row.sort_values()[::-1][0:k]
        top_k_list.setdefault(key, list(basket.index))
    return top_k_list


def construct_dict(data):
    user_dict = {}
    item_dict = {}
    for row in data.itertuples():
        user_dict.setdefault(row[1], []).append(row[2])
        item_dict.setdefault(row[2], []).append(row[1])
    return user_dict, item_dict


def evaluation(topk_dict, test_dict, k_val, test_dict_num):
    relevant_num = 0
    topk_dict_num = len(topk_dict) * k_val

    for key, value in topk_dict.items():
        #rank = 0
        if key in test_dict:
            overlap_set = set(value).intersection(test_dict[key])
            row_overlap_num = len(overlap_set)
            relevant_num += row_overlap_num
            #for i in overlap_set:
            #    rank += ((i/value.index(i)) for i in overlap_set)
                    
                    
    print('relevant num:' + str(relevant_num))
    print('topk_dict_num:' + str(topk_dict_num))
    print('test_dict_num:' + str(test_dict_num))
    
    precision = relevant_num / topk_dict_num
    recall = relevant_num / test_dict_num
    return precision, recall

def filter_rating(data_matrix,data_dict):
    filter_matrix = data_matrix.copy()
    for key,value in data_dict.items():
        filter_matrix.loc[[key],value] = 0
    return filter_matrix

##
header = ['userId', 'itemId', 'ratings']
train_data = pd.read_csv('/home/elics-lee/acdamicSpace/dataset/ml-1m/graph/train.csv', sep=' ',names=header)
test_data = pd.read_csv('/home/elics-lee/acdamicSpace/dataset/ml-1m/graph/test.csv', sep=' ', names=header)
topk_value = 30
user_num = 6040

emb_file = '/home/elics-lee/acdamicSpace/dataset/ml-1m/emb/ml.emb'

#emb_file = 'emb/filmtrustK10WalkL30.emb'
#emb_file = 'emb/filmtrustK10WalkL120.emb'
#emb_file = 'emb/filmtrustK10WalkL160.emb'

#emb_file = 'emb/filmtrustK10.emb'
#emb_file = 'emb/filmtrustK30.emb'
#emb_file = 'emb/filmtrustK60.emb'
#emb_file = 'emb/filmtrustK128.emb'
#emb_file = 'emb/filmtrustK15.emb'
#emb_file = 'emb/filmtrustK8.emb'
sim_matrix, label_index = load_DataFrame(emb_file)
item_matrix, user_matrix = matrix_split(sim_matrix, user_num)
user_dict_train, item_dict_train = construct_dict(train_data)
user_dict_test, item_dict_test = construct_dict(test_data)

filter_user_matrix = filter_rating(user_matrix,user_dict_train)
filter_item_matrix = filter_rating(item_matrix,item_dict_train)

user_top_list = top_k_list(filter_user_matrix,topk_value)
item_top_list = top_k_list(filter_item_matrix,topk_value)

user_precision, user_recall = evaluation(user_top_list,user_dict_test,topk_value,len(test_data))
item_precision, item_recall = evaluation(item_top_list,item_dict_test,topk_value,len(test_data))

print('User-based embeding Precision: %.5f' %(user_precision))
print('User-based embeding recall: %.5f' %(user_recall))      
print('Item-based embeding Precision: %.5f' %(item_precision))
print('Item-based embeding recall: %.5f' %(item_recall))      
