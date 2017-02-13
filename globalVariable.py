"""
    设定全局变量

    Authors: Elics Lee(yunzhan)

    Date: 12/21 2016

    License: BSD 3 clause
"""
from loadData import loadRating
#from loadData import loadTrust
#from loadData import averageRating
from loadData import delRedundatInfor

'''
    trusting信息无论是否采用训练集和测试集的方式都是一样的
'''
trust_path = '/home/li/acadamic/dataset/FilmTrust/trust.txt'

#trust_dict = loadTrust(trust_path,split_sig='\t')
trust_dict = delRedundatInfor(trust_path,split_sig=' ')

######################### 完整的数据集作为训练集和测试集 #######################

#rat_path = '/home/li/acadamic/dataset/ciao/ciao_rating.txt'
#trust_path = '/home/li/acadamic/dataset/ciao/ciao_trust.txt'
#rat_path = '/home/li/acadamic/dataset/epinions/rating.txt'
#trust_path = '/home/li/acadamic/dataset/epinions/trust.txt'
rat_path = '/home/li/acadamic/dataset/FilmTrust/ratings.txt'
user_item_dict = loadRating(rat_path,main_keys='user',split_sig=' ')
item_user_dict = loadRating(rat_path,main_keys='item',split_sig=' ')


#####################交叉验证的数据集分割#######################

traing_data_path = '/home/li/acadamic/dataset/FilmTrust/FilmTrust_train.data'
testing_data_path = '/home/li/acadamic/dataset/FilmTrust/FilmTrust_test.data'

#训练集的user_item_dict
#训练集的item_user_dict
traing_user_item_dict = loadRating(traing_data_path,main_keys='user',split_sig='\t')
traing_item_user_dict = loadRating(traing_data_path,main_keys='user',split_sig='\t')
#训练集的user_item_dict
#训练集的item_user_dict
testing_user_item_dict = loadRating(testing_data_path,main_keys='user',split_sig='\t')
testing_item_user_dict = loadRating(testing_data_path,main_keys='item',split_sig='\t')























