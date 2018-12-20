# 3. 机器学习问题解决思路
# 上面带着代价走马观花过了一遍机器学习的若干算法，下面我们试着总结
# 在拿到一个实际问题的时候，如果着手使用机器学习算法去解决问题，其
# 中的一些注意点以及核心思路。主要包括以下内容：
#
# 拿到数据后怎么了解数据(可视化)
# 选择最贴切的机器学习算法
# 定位模型状态(过/欠拟合)以及解决方法
# 大量极的数据的特征分析与可视化
# 各种损失函数(loss function)的优缺点及如何选择
#
# 多说一句，这里写的这个小教程，主要是作为一个通用的建议和指导方案，
# 你不一定要严格按照这个流程解决机器学习问题。
# ---------------------

# 3.1 数据与可视化
# 我们先使用scikit-learn的make_classification函数来生产一份分类数据，
# 然后模拟一下拿到实际数据后我们需要做的事情。

import  numpy as np
from sklearn.datasets import make_classification
import  matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

X,y = make_classification(1000,
                          n_features=20,
                          n_informative=2,
                          n_redundant=2,
                          n_classes=2,
                          random_state=0)
columns = list(range(20)) + ["class"]
df = DataFrame(np.hstack((X,y[:,None])),
               columns= columns)
sns.pairplot(df[:50],
             vars = [8,11,12,14,19],
             hue = "class",
             height = 1.5,
             diag_kind = 'hist',
             palette = 'husl')
plt.figure(figsize=(12,10))
cor = df.corr()
cor = np.tril(cor,-1)
# cor = abs(cor)
# cmap = sns.cubehelix_palette(start = 0, rot = 1, gamma=0.8, as_cmap = True)
# cmap = sns.cubehelix_palette()
# cmap = sns.light_palette("seagreen", reverse=False)
# cmap = sns.light_palette("gray")
# cmap = sns.light_palette((8, 10, 20), input="husl")
# cmap 取值范围：支持matplotlib库中colormap，支持以下：
# "PiYG","PRGn","BrBG","PuOr","RdGy","RdBu","RdYIBu",
# "RdYIGn","Spectral","coolwarm","bwr","seismic"
ax = sns.heatmap(cor,
                 vmin = -1,
                 vmax = 1,
                 annot = False,
                 cmap = "PiYG",
                 mask = (cor==0),
                 xticklabels = False,
                 yticklabels = False)
for x,s in enumerate(columns):
    # print(x,s)
    plt.text(x+0.29,
             x+0.65,
             "%s"%s)
plt.show()

# 相关性图很好地印证了我们之前的想法，可以看到第11维特征和第14维特征和类别有极强的相关性，
# 同时它们俩之间也有极高的相关性。而第12维特征和第19维特征却呈现出极强的负相关性。
# 强相关的特征其实包含了一些冗余的特征，而除掉上图中颜色较深的特征，其余特征包含的信息量
# 就没有这么大了，它们和最后的类别相关度不高，甚至各自之间也没什么相关性。
# ---------------------
# 作者：寒小阳
# 来源：CSDN
# 原文：https://blog.csdn.net/han_xiaoyang/article/details/50469334
# 版权声明：本文为博主原创文章，转载请附上博文链接！