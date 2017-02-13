import numpy as np
import math
import similarity as sl
import time
from similarity import sim_pearson

#################### 导入全局变量 ##################
from globalVariable import trust_dict
from globalVariable import user_item_dict
from globalVariable import item_user_dict

##################构造P矩阵############################
def p_matrix_cons(matrix_size):
    user_array = np.zeros((matrix_size,matrix_size))
    #print(length)
    for i in range(matrix_size):
        if(i+1 in trust_dict):
            for j in trust_dict[i+1]:
                user_array[i][j-1] = 1/len(trust_dict[i+1])
    return user_array

##################构造对角矩阵###########################
def diagonal_phi(goal_item,step_num):
    matrix_size = max(user_item_dict)
    diagonal_matrix = np.zeros((matrix_size,matrix_size))

    for i in range(matrix_size):
        max_sim = 0
        for item in user_item_dict[i+1].keys():
            ## 求每个用户所评过的项目集中与目标项目最相似的
            i_item_dict = item_user_dict[goal_item]
            j_item_dict = item_user_dict[item]
            local_sim = sim_pearson(i_item_dict,j_item_dict)
#            if(local_sim != 0):
#                print(local_sim)
            if(local_sim>max_sim):
                max_sim = local_sim
        if(max_sim>=1):
            print(max_sim)

        diagonal_matrix[i][i] = 1-max_sim/(1+math.exp(-step_num/2))
        ## print(diagonal_matrix[i][i])

    return diagonal_matrix

#################### random walk phi i,j,k 为0的单位矩阵情况##########
def randomWalk(goal_item,step_num):
    matrix_size = max(user_item_dict)

    diagonal_matrix = np.identity(matrix_size)

    return diagonal_matrix

#################计算得出矩阵P*##########################
def p_star_matrix(goal_item,step_num=1,method=randomWalk):
    matrix_size = max(user_item_dict)
    p_array = p_matrix_cons(matrix_size)
    step_matrix = np.eye(matrix_size)
    p_star_matrix = np.zeros((matrix_size,matrix_size))

    start = time.clock()
    for i in range(step_num):
        diag_phi = method(goal_item,i+1)
        print("++++++++++++++++++"+str(i)+"++++++++++++++++++++")
        print(p_array)
        new_matrix = np.dot(diag_phi,p_array)
        step_matrix = np.dot(step_matrix,new_matrix)
        p_star_matrix = p_star_matrix+step_matrix
    end = time.clock()
    print(end-start)

    return p_star_matrix

##########################矩阵进行归一化处理############################
def p_normalization(pstart):
    size = len(pstart)
    c=np.eye(size)
    for row in range(size):
        s=sum(pstart[row])
        if(s!=0):
            c[row][row]=1.0/s
        else:
            c[row][row]=0
            # print sum(pstart[0])
    pn = np.mat(pstart)*np.mat(c)
    return pn

######################### initial R item ############################
def initRatingVector(goal_item):
    user_num = max(user_item_dict)
    init_rating_vector = np.zeros((user_num,1))
    for i in range(user_num):
        print(i)
        if((i+1) in item_user_dict[goal_item]):
            init_rating_vector[i] = item_user_dict[goal_item][i+1]
        else:
            print("+++++++++++++++")
            if((i+1) in user_item_dict):
                user_rating_vector = user_item_dict[i+1]
                init_rating_vector[i] = sl.maxSimRating(user_rating_vector,goal_item)
            else:
                init_rating_vector[i] = .0

    return init_rating_vector

########################## initialItem ##############################
def initialItems(goal_item):
    user_num = max(user_item_dict)
    itemScores = item_user_dict[goal_item]
    ri = np.zeros((user_num,1))

    for i in range(user_num):
        itemScores.setdefault(i+1,0)
        if(itemScores[i+1]!=0):
            ri[i] = itemScores[i+1]

    print(ri)
    return ri

########################### source item ##############################
def sourceItems(goal_item):
    user_num = max(user_item_dict)
    #itemScores = item_user_dict[goal_item]
    ri = np.zeros((user_num,1))

    for i in range(user_num):
        #ri[i] = sum(itemScores.values())/len(itemScores)
        ri[i] = 3.0


    print(ri)
    return ri

######################## exit programming ############################
def exit_programming(pNormalization,initRating):
    user_num = max(user_item_dict)
    Sum=0.0
    times = 10000
    e=0.0001
    resultNew=10.0
    resultOld=0.0
    ri = np.zeros((user_num,1))
    for i in range(1,times+2):
        if i==1:
            aver_r=initRating
        else:
            resultOld=Sum/(i-1)
            ri=pNormalization*np.mat(initRating)
            initRating=ri
            aver_r=(ri+(aver_r*(i-1)))/i
            temp=math.pow(np.linalg.norm((aver_r-ri),2),2)
            Sum+=temp
            resultNew=Sum/i
            print ('time '+str(i))
        if math.fabs(resultNew-resultOld)<e:
            print(i)
            return ri
    return ri

#######################RMSE计算######################################
def compute_rmse(peridict_ri,ri):
    length = len(peridict_ri)
    num_rui = 0
    sum_r = 0.0
    for i in range(length):
        if(ri[i]!=0):
            num_rui = num_rui+1
            sum_r = sum_r+math.pow((peridict_ri[i]-ri[i]),2)
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


########################### 执行完整过程 ############################
def excute(item):
    ## init ri
    #init_ri = initRatingVector(item)
    ## source rating vector
    sour_ri = sourceItems(item)
    ## coumpute normalazition matrix hat p
    #item_i_pStar_matrix = p_star_matrix(item,6)
    item_i_pStar_matrix = p_matrix_cons(max(user_item_dict))
    #normal_matrix = p_normalization(item_i_pStar_matrix)
    normal_matrix = np.mat(item_i_pStar_matrix)

    ##
    prediction_ri = exit_programming(normal_matrix,sour_ri)

    rmse = compute_rmse(prediction_ri ,sour_ri)
    mae = compute_mae(prediction_ri ,sour_ri)

    print(prediction_ri,mae)
    return rmse

############################## 交叉验证 #################################
#def

def main():
    return 0

################################test##################################
def test():
    length = 0
    for k,v in item_user_dict.items():
        temp = len(v)
        if(length<temp):
            length = temp
            #result = k
    return len(user_item_dict)


if __name__=="__main__":
    main()



















