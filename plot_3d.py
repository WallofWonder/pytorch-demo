import matplotlib.pyplot as plt
import numpy as np

# 求向量积(outer()方法又称外积)
x = np.outer(np.linspace(-2, 2, 30), np.ones(30))
# 矩阵转置
y = x.copy().T
# 数据z
z = np.cos(x ** 2 + y ** 2)
# 绘制曲面图
fig = plt.figure()
ax = plt.axes(projection='3d')
# 调用plot_surface()函数
ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
plt.show()
