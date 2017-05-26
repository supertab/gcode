# 映射函数, 从一个连续区间, 映射到离散区间
# 连续区间: [-10.22, 20.31] 离散区间[0, 255]
def map_a2b(num, a=[-10.22, 20.31], b=[0, 255]):
    min_a, max_a = min(a), max(a)
    min_b, max_b = min(b), max(b)
    range_a = max_a - min_a
    range_b = max_b - min_b
    # mapping num from a to b
    tmp =  ((num-min_a)/range_a) * range_b + min_b
    # round
    return int('%.0f'%tmp)
