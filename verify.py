"""
 Function： 能够对trust wallk training 训练过后的模型进行结果验证

 Author: Elics Lee(yunzhan)

 Date: 12/21 2016

"""
import numpy as np
import evalutionCompute as ec
import training as traing

# 引入testing data作为全局变量
from globalVariable import item_user_dict
from globalVariable import user_item_dict
from globalVariable import testing_item_user_dict
from globalVariable import testing_user_item_dict

################### testing data rating vector#####################
def testRatingVector(goal_item):
    user_num = max(testing_user_item_dict)
    itemScores = testing_item_user_dict[goal_item]
    ri = np.zeros((user_num,1))

    for i in range(user_num):
        itemScores.setdefault(i+1,0)
        if(itemScores[i+1]!=0):
            ri[i] = itemScores[i+1]

    print(ri)
    return ri

####################### source item ##########################
def sourceItems(goal_item):
    user_num = max(user_item_dict)
    itemScores = item_user_dict[goal_item]
    ri = np.zeros((user_num,1))

    for i in range(user_num):
        ri[i] = sum(itemScores.values())/len(itemScores)


    print(ri)
    return ri

###################### 训练模型预测比对 ####################
def verifyResult(item):
    #evalution_vector = {}
    predict_ri = traing.excute(item)
    ri = testRatingVector(item)
    rmse = ec.compute_rmse(predict_ri,ri)
    mae = ec.compute_mae(predict_ri,ri)
    print(rmse,mae)


