######################load dataset####################
def loadRating(path, main_keys="item", split_sig='\t'):
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


#######################load MovieLens dataset####################
#def loadMoiveLens(path_item, path_data, main_keys="item", split_sig='\t'):
#    # 获得item Id和 UserId 构建存储项目、用户、评分之间关系的数据结构
#    movies = {}
#    for line in open(path_item):
#        (id, title) = line.split('|')[0:2]
#        movies[id] = title
#
#    setofItem = {}
#    count = 0
#    for line in open(path_data):
#        (userId, itemId, rating, time) = line.split(split_sig)
#        if (main_keys == "item"):
#            setofItem.setdefault(int(itemId), {})
#            setofItem[int(itemId)][int(userId)] = float(rating)
#            count = count + 1
#        else:
#            setofItem.setdefault(int(userId), {})
#            setofItem[int(userId)][int(itemId)] = float(rating)
#            count = count + 1
#
#    print(len(setofItem))
#    return setofItem
#

#####################load Turst##########################
def loadTrust(path, split_sig='\t'):
    trustmatrix = {}
    ###统计表中有多少个用户
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


#########################用二维数组存储Rating矩阵#################
def rating_nparray():
    i = 0
    print(i)
    return 0


################### user average rating ####################
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

##################### 翻转user item dict ######################
def transformPrefs(prefs):
    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})

            # Flip item and person
            result[item][person] = prefs[person][item]
    return result
