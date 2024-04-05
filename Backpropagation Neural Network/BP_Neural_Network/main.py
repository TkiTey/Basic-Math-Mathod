import bp_fuc
import numpy as np
import matplotlib.pyplot as plt


X = list(np.arange(-1, 1.1, 0.1))  # -1~1.1 步长0.1增加
D = [-0.96, -0.577, -0.0729, 0.017, -0.641, -0.66, -0.11, 0.1336, -0.201, -0.434, -0.5,
     -0.393, -0.1647, 0.0988, 0.3072, 0.396, 0.3449, 0.1816, -0.0312, -0.2183, -0.3201]
A = X + D  # 数据合并 方便处理
patt = np.array([A] * 2)  # 2*42矩阵
# 创建神经网络，21个输入节点，13个隐藏层节点，21个输出层节点
bp = bp_fuc.BP_NN(21, 13, 21)
# 训练神经网络
bp.train(patt)
# 测试神经网络
d = bp.test(patt)
# 查阅权重值
bp.weights()


plt.plot(X, D, label="source data")  # D为真实值
plt.plot(X, d, label="predict data")  # d为预测值
plt.legend()
plt.show()
