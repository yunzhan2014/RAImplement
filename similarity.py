import math
from math import sqrt

###### 导入全局量
# from globalVariable import item_user_dict
# from globalVariable import user_item_dict

# 导入训练集对应的dict
# from globalVariable import traing_item_user_dict
from globalVariable import traing_user_item_dict
from loadData import transformPrefs


def sim_pearson(i_item_dict, j_item_dict):
    # 找到对ij都进行过评分的用户集合
    n = len(traing_user_item_dict)
    num_uij = 0
    sum_i = 0
    sum_j = 0
    sum_iSq = 0
    sum_jSq = 0
    pSum = .0
    # 计算i的累加和与累加平方和
    for key, value in i_item_dict.items():
        sum_i += value
        sum_iSq += math.pow(value, 2)
    # 计算j的累加和与累加平方和
    for key, value in j_item_dict.items():
        sum_j += value
        sum_jSq += math.pow(value, 2)
    # 计算i,j共同评论的用户数和项目评分相乘的和
    for key, value in i_item_dict.items():
        if (key in j_item_dict):
            num_uij = num_uij + 1
            pSum += i_item_dict[key] * j_item_dict[key]
    # 计算分子
    num = pSum - (sum_i * sum_j / n)
    # 虽然item_user_dict里有项目，但为0
    if sum_j == 0 or sum_jSq == 0:
        return 0
    # 计算分母
    den = ((sum_iSq - pow(sum_i, 2) / n) * (sum_jSq - pow(sum_j, 2) / n)) ** .5
    # 分母为0，则返回1
    if den == 0:
        return 1.0  # *(1.0/(1+math.exp(-num_uij/2)))
    return (num / den)  # *(1.0/(1+math.exp(-num_uij/2)))


def cos(i_dict, j_dict):
    dot_product = 0
    i_model_long = 0
    j_model_long = 0
    result = 0
    for i in range(1, len(i_dict)):
        dot_product += i_dict[i] * j_dict[i]
        i_model_long += math.pow(i_dict[i], 2)
        j_model_long += math.pow(j_dict[i], 2)
    print(dot_product, j_model_long, i_model_long)
    if (dot_product == 0):
        return result
    else:
        return dot_product / (math.sqrt(i_model_long * j_model_long))


def adjusted_cos(rating_dict, i, j):
    # Get list of ratings i and j
    si = []
    for item in rating_dict[i]:
        if item in rating_dict[j]:
            si.append(item)

    return 0


# Returns a distance-based similarity score for person1 and person2
def sim_distance(prefs, person1, person2):
    # Get the list of shared_items
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]: si[item] = 1

    # if they have no ratings in common, return 0
    if len(si) == 0: return 0

    # Add up the squares of all the differences
    sum_of_squares = sum([pow(prefs[person1][item] - prefs[person2][item], 2)
                          for item in prefs[person1] if item in prefs[person2]])

    return 1 / (1 + sum_of_squares)


# Returns the Pearson correlation coefficient for p1 and p2
def sim_pearson2(prefs, p1, p2):
    # Get the list of mutually rated items
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]: si[item] = 1

    # if they are no ratings in common, return 0
    if len(si) == 0: return 0

    # Sum calculations
    n = len(si)

    # Sums of all the preferences
    sum1 = sum([prefs[p1][it] for it in si])
    sum2 = sum([prefs[p2][it] for it in si])

    # Sums of the squares
    sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])
    sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])

    # Sum of the products
    pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])

    # Calculate r (Pearson score)
    num = pSum - (sum1 * sum2 / n)
    den = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
    if den == 0: return 0

    r = num / den

    return r


# Returns the best matches for person from the prefs dictionary.
# Number of results and similarity function are optional params.
def topMatches(prefs, person, n=5, similarity=sim_pearson2):
    scores = [(similarity(prefs, person, other), other)
              for other in prefs if other != person]
    scores.sort()
    scores.reverse()


def calculateSimilarItems(prefs, n=10):
    # Create a dictionary of items showing which other items they
    # are most similar to.
    result = {}
    # Invert the preference matrix to be item-centric
    itemPrefs = transformPrefs(prefs)
    c = 0
    for item in itemPrefs:
        # Status updates for large datasets
        c += 1
        if c % 100 == 0: print("%d / %d" % (c, len(itemPrefs)))
        # Find the most similar items to this one
        scores = topMatches(itemPrefs, item, n=n, similarity=sim_pearson2)
        result[item] = scores
    return result


def main():
    rating_dict = {'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
                                 'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
                                 'The Night Listener': 3.0},
                   'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
                                    'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
                                    'You, Me and Dupree': 3.5},
                   'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
                                        'Superman Returns': 3.5, 'The Night Listener': 4.0},
                   'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
                                    'The Night Listener': 4.5, 'Superman Returns': 4.0,
                                    'You, Me and Dupree': 2.5},
                   'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                                    'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
                                    'You, Me and Dupree': 2.0},
                   'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                                     'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
                   'Toby': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 'Superman Returns': 4.0}}
    i_dict = {1: 2.0, 2: 3.0, 3: 4.0, 4: 4.0, 5: 5.0}
    j_dict = {1: 3.0, 2: 4.0, 3: 5.0, 4: 3.0, 5: 4.0}
    cos_rating = cos(i_dict, j_dict)
    print(cos_rating)
    adjusted_cos_score = adjusted_cos(rating_dict, 'Jack Matthews', 'Toby')
    print('adjusted cosine score:', adjusted_cos_score)
