import numpy as np
import random


# 定义激活函数及其导数
def sigmoid(x):
    return np.tanh(x)


def DS(x):
    return 1 - (np.tanh(x)) ** 2


# 生成一个m*n矩阵，并且设置默认零矩阵
def makematrix(m, n, fill=0.0):
    a = []
    for i in range(m):
        a.append([fill] * n)  # 列表1*n会得到一个新列表，新列表元素为列表1元素重复n次。[fill]*3==[fill fill fill]
    return np.array(a)


# 生成区间[a,b]内的随机数
def random_number(a, b):
    return (b - a) * random.random() + a  # random.random()随机生成[0,1)内浮点数


class BP_NN:
    def __init__(self, num_in, num_hidden, num_out):
        # 设置各层的节点数
        # +1为了做调整
        self.num_in = num_in + 1  # 输入层
        self.num_hidden = num_hidden + 1  # 隐藏层
        self.num_out = num_out  # 输出层
        # 激活结点
        self.active_in = np.array([-1.0] * self.num_in)
        self.active_hidden = np.array([-1.0] * self.num_hidden)
        self.active_out = np.array([1.0] * self.num_out)
        # 初始化权重矩阵
        # 创建
        self.weight_in = makematrix(self.num_in, self.num_hidden)
        self.weight_out = makematrix(self.num_hidden, self.num_out)
        # 对权重矩阵weight赋初值
        for i in range(self.num_in):  # 对weight_in矩阵赋初值
            for j in range(self.num_hidden):
                self.weight_in[i][j] = random_number(0.1, 0.1)
        for i in range(self.num_hidden):  # 对weight_out矩阵赋初值
            for j in range(self.num_out):
                self.weight_out[i][j] = random_number(0.1, 0.1)
        # 偏差,与上面+1照应
        for j in range(self.num_hidden):
            self.weight_in[0][j] = 0.1
        for j in range(self.num_out):
            self.weight_out[0][j] = 0.1

        # 建立动量因子（矩阵），初始化
        self.ci = makematrix(self.num_in, self.num_hidden)  # num_in*num_hidden 矩阵
        self.co = makematrix(self.num_hidden, self.num_out)  # num_hidden*num_out矩阵

    # 信号正向传播
    def update(self, inputs):
        if len(inputs) != (self.num_in - 1):
            raise ValueError("与输入层结点数不符")
        # 数据输入 输入层
        self.active_in[1:self.num_in] = inputs
        # 数据在隐藏层处理
        self.sum_hidden = np.dot(self.weight_in.T, self.active_in.reshape(-1, 1))  # 叉乘
        # .T操作是对于array操作后的数组进行转置操作
        # .reshape(x,y)操作是对于array操作后的数组进行重新排列成一个x*y的矩阵，参数为负数表示无限制，如(-1,1)转换成一列的矩阵
        self.active_hidden = sigmoid(self.sum_hidden)  # active_hidden[]是处理完输入数据之后处理，作为输出层的输入数据
        self.active_hidden[0] = -1
        # 数据在输出层处理
        self.sum_out = np.dot(self.weight_out.T, self.active_hidden)
        self.active_out = sigmoid(self.sum_out)
        # 返回输出层结果
        return self.active_out

    #误差反向传播
    def errorbackpropagate(self, targets, lr, m):  # lr 学习效率
        if self.num_out == 1:
            targets = [targets]
        if len(targets) != self.num_out:
            raise ValueError("与输出层结点数不符")
        # 误差
        error = (1 / 2) * np.dot((targets.reshape(-1, 1) - self.active_out).T,
                                 (targets.reshape(-1, 1) - self.active_out))

        # 输出层 误差信号
        self.error_out = (targets.reshape(-1, 1) - self.active_out) * DS(self.sum_out)  # DS(self.active_out)
        # 隐层 误差信号
        self.error_hidden = np.dot(self.weight_out, self.error_out) * DS(self.sum_hidden)  # DS(self.active_hidden)

        # 更新权值
        # 隐层
        self.weight_out = self.weight_out + lr * np.dot(self.error_out,self.active_hidden.reshape(1, -1)).T + m * self.co
        self.co = lr * np.dot(self.error_out, self.active_hidden.reshape(1, -1)).T
        # 输入层
        self.weight_in = self.weight_in + lr * np.dot(self.error_hidden,self.active_in.reshape(1, -1)).T + m * self.ci
        self.ci = lr * np.dot(self.error_hidden, self.active_in.reshape(1, -1)).T

        return error

    #训练函数
    def train(self, pattern, itera=100, lr=0.2, m=0.1):
        for i in range(itera):
            error = 0.0  # 每一次迭代将error置0
            for j in pattern:  # j为传入数组的第一维数据
                inputs = j[0:self.num_in - 1]  # 根据输入层结点的个数确定传入结点值的个数
                targets = j[self.num_in - 1:]  # 剩下的结点值作为输出层的值
                self.update(inputs)  # 正向传播 更新了active_out
                error = error + self.errorbackpropagate(targets, lr, m)  # 误差反向传播 计算总误差
            if i % 10 == 0:
                print("########################误差 %-.5f ######################第%d次迭代" % (error, i))

    # ? 测试
    def test(self, patterns):
        for i in patterns:  # i为传入数组的第一维数据
            print(i[0:self.num_in - 1], "->", self.update(i[0:self.num_in - 1]))
        return self.update(i[0:self.num_in - 1])  # 返回测试结果，用于作图

    # ? 权值
    def weights(self):
        print("输入层的权值：")
        print(self.weight_in)
        print("输出层的权值：")
        print(self.weight_out)
