
# n, m, c = input().split()
#
# shou_zhu = dict()
# for x in range(1, n):
#     line_input = input().split()
#     shou_zhu.setdefault(line_input[0], line_input[1:])




def find(str):
    two_char_list = dict()
    result_list = list()
    index = 0
    for char in str:
        if char not in two_char_list:
            two_char_list.setdefault(index, 1)
        else:
            two_char_list[index] = two_char_list[index]+1
        if two_char_list[index] == 2:
            result_list.append(char)
    result = [x  for x in two_char_list.items() if x == 2]
    return len(result)


def search_index(str_input):
    if str_input is None:
        return None
    index_result = 0
    factor_vector = [1 + 25 + 25 * 25 + 25 * 25 * 25, 1 + 25 + 25 * 25, 1 + 25 * 25, 1]
    count = 0
    for x in str_input:
        char_str = ord(x) - 97
        index_result += char_str * factor_vector[count]
        count += 1
    return index_result


str_input = input()
print(search_index(str_input))


def find_prime_pair(num):
    prime_coll = dict()
    for x in range(2, num+1):
        if x not in prime_coll:
            prime_coll[x] = 1
            k = x*2
            while k<= num:
                prime_coll[x] = 0
                k += x
    count = 0
    for j in prime_coll:
        if j == 1 and prime_coll[num-1]== 1:
            count += 1
        elif j == num:
            count += 1
    return count


city_num,