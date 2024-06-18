import numpy as np
import matplotlib.pyplot as plt

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

def gradientDescendent(f, p, learning_rate=0.01, max_loops=1000, tol=1e-6, dump_period=100):
    # 將普通數值轉換為 Value 類
    p_values = [Value(x) for x in p]

    for i in range(max_loops):
        # 計算損失
        loss = f(p_values)

        # 清零梯度
        for v in p_values:
            v.grad = 0

        # 反向傳播計算梯度
        loss.backward()

        # 更新參數
        for v in p_values:
            v.data -= learning_rate * v.grad

        if i % dump_period == 0 or i == max_loops - 1:
            print(f'Iteration {i}, Loss: {loss.data}, Params: {[v.data for v in p_values]}')

        # 檢查收斂
        if loss.data < tol:
            break

    return [v.data for v in p_values]

# 測試案例 gdArray.py
def test_gdArray():
    def f(p):
        [x, y, z] = p
        return (x-1)**2 + (y-2)**2 + (z-3)**2

    p = [0.0, 0.0, 0.0]
    gradientDescendent(f, p)

# 測試案例 gdRegression.py
def test_gdRegression():
    x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    y = np.array([1.9, 3.1, 3.9, 5.0, 6.2], dtype=np.float32)

    def predict(a, xt):
        return a[0] + a[1] * xt

    def MSE(a, x, y):
        total = 0
        for i in range(len(x)):
            total += (y[i] - predict(a, x[i]))**2
        return total

    def loss(p):
        return MSE(p, x, y)

    p = [0.0, 0.0]
    plearn = gradientDescendent(loss, p, max_loops=3000, dump_period=100)

    # Plot the graph
    y_predicted = list(map(lambda t: plearn[0] + plearn[1] * t, x))
    print('y_predicted=', y_predicted)
    plt.plot(x, y, 'ro', label='Original data')
    plt.plot(x, y_predicted, label='Fitted line')
    plt.legend()
    plt.show()

# 選擇要運行的測試案例
if __name__ == "__main__":
    print("Running gdArray test case:")
    test_gdArray()
    
    print("\nRunning gdRegression test case:")
    test_gdRegression()
