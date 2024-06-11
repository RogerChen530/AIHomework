import numpy as np
from scipy.optimize import linprog

# 目標函數係數
c = np.array([-3, -2, -5])

# 限制條件係數矩陣
A = np.array([
    [1, 1, 0],
    [2, 0, 1],
    [0, 1, 2]
])

# 限制條件右側值
b = np.array([10, 9, 11])

# 變數的界限
x_bounds = (0, None)
y_bounds = (0, None)
z_bounds = (0, None)

# 使用 linprog 函數求解線性規劃問題
res = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, y_bounds, z_bounds], method='highs')

# 打印結果
print("Optimal Solution:")
print(f"x = {res.x[0]}")
print(f"y = {res.x[1]}")
print(f"z = {res.x[2]}")
print(f"Maximized Objective Function Value = {-res.fun}")  # linprog 返回的是最小值，所以需要取負號
