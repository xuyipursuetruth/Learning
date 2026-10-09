import numpy as np

import numpy as np

def make_linear_separable(n_samples=200, n_features=2, margin=0.5, random_state=None):
    """
    生成线性可分数据集
    margin: 样本与分离超平面的最小距离，越大越容易分
    """
    rng = np.random.RandomState(random_state)

    # 1. 随机生成分离超平面的法向量 w 和偏置 b
    w = rng.randn(n_features)
    w /= np.linalg.norm(w)          # 归一化，方便控制距离
    b = rng.randn() * 0.5

    X, y = [], []
    while len(X) < n_samples:
        x = rng.uniform(-3, 3, n_features)
        # 点到超平面的有符号距离
        dist = x @ w + b
        if dist > margin:           # 正类，且离超平面足够远
            X.append(x); y.append(1)
        elif dist < -margin:        # 负类
            X.append(x); y.append(-1)
    return np.array(X), np.array(y)

# 原始形式
class Perceptron_v1():
    def __init__(self,x_dim=2,lr=1,maxiter=1000):
        self.w = np.zeros(x_dim)
        self.b = 0
        self.lr = lr
        self.maxiter = maxiter

    def learning(self,x,y):
        # x [N,D]
        # y [N]
        for _ in range(self.maxiter):
            misclassify = False
            for i in range(y.shape[0]):
                if (y[i]*self.w.dot(x[i])) <= 0:
                    self.w = self.w + self.lr*y[i]*(x[i])
                    self.b = self.b + self.lr*y[i]
                    print(self.w,self.b,i)
                    misclassify = True       
            if not misclassify:
                break

# 对偶形式
class Perceptron_v2():
    def __init__(self,x_dim=2,lr=None,maxiter=1000):
        self.alpha = None
        self.b = 0
        self.lr = lr
        self.maxiter = maxiter
        self.gram = None

    def learning(self,x:np.ndarray,y:np.ndarray):
        # x [N,D]
        # y [N]
        self.gram = x @ x.T
        self.alpha = np.zeros(x.shape[0])

        for _ in range(self.maxiter):
            misclassify = False
            for i in range(y.shape[0]):
                if y[i]*(np.sum(self.alpha*y*self.gram[:,i])+self.b) <= 0:
                    self.alpha[i] += self.lr
                    self.b += self.lr*y[i]
                    misclassify = True
            if not misclassify:
                break
        
        
if __name__ == "__main__":
    data_num = 5
    train_x = np.random.randint(0,10,size=(data_num,2))
    train_y = np.random.choice((-1,1),size=data_num)
    print(train_x,train_y)
    model_1 = Perceptron_v1()
    model_1.learning(train_x,train_y)
