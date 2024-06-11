import random

citys = [
    (0, 3), (0, 0),
    (0, 2), (0, 1),
    (1, 0), (1, 3),
    (2, 0), (2, 3),
    (3, 0), (3, 3),
    (3, 1), (3, 2)
]

def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def pathLength(p):
    dist = 0
    plen = len(p)
    for i in range(plen):
        dist += distance(citys[p[i]], citys[p[(i + 1) % plen]])
    return dist

def random_swap(p):
    # 隨機交換兩個城市的位置
    new_path = p.copy()
    idx1, idx2 = random.sample(range(len(p)), 2)
    new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]
    return new_path

def hill_climbing(initial_path, max_iter):
    current_path = initial_path
    current_length = pathLength(current_path)
    for _ in range(max_iter):
        new_path = random_swap(current_path)
        new_length = pathLength(new_path)
        if new_length < current_length:
            current_path = new_path
            current_length = new_length
    return current_path, current_length

if __name__ == "__main__":
    initial_path = [i for i in range(len(citys))]
    random.shuffle(initial_path)
    print("初始路徑:", initial_path)
    print("初始路徑長度:", pathLength(initial_path))
    
    final_path, final_length = hill_climbing(initial_path, max_iter=1000)
    print("最終路徑:", final_path)
    print("最終路徑長度:", final_length)
