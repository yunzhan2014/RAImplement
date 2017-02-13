"""
    Funtion: 计算推荐系统评估参数

    Author: Elics Lee(yunzhan)

    Date: 12/21 2016

"""
import numpy as np
import math

#######################RMSE计算######################################
def compute_rmse(peridict_ri,ri):
    length = len(peridict_ri)
    num_rui = 0
    sum_r = 0.0
    for i in range(length):
        if(ri[i]!=0):
            if(peridict_ri[i]!=0):
                print(peridict_ri[i])
            num_rui = num_rui+1
            sum_r = sum_r+math.pow((peridict_ri[i]-ri[i]),2)
    print(num_rui)
    return math.sqrt(sum_r/num_rui)

######################### MAE计算 ####################################
def compute_mae(perdict_ri,ri):
    length = len(perdict_ri)
    common_rating_num = 0
    sum_r = 0.0
    for i in range(length):
        if(ri[i] != 0):
            common_rating_num+=1
            sum_r += np.absolute(perdict_ri[i]-ri[i])
    return sum_r/common_rating_num

