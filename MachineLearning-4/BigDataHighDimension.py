# 3.3 关于大数据样本集和高维特征空间
# 我们在小样本的toy dataset上，怎么捣鼓都有好的方法。但是当数据量和特征样本空间膨胀非常
# 厉害时，很多东西就没有那么好使了，至少是一个很耗时的过程。举个例子说，我们现在重新生
# 成一份数据集，但是这次，我们生成更多的数据，更高的特征维度，而分类的类别也提高到5。
# 3.3.1 大数据情形下的模型选择与学习曲线
# 在上面提到的那样一份数据上，我们用LinearSVC可能就会有点慢了，我们注意到机器学习算法
# 使用图谱推荐我们使用SGDClassifier。其实本质上说，这个模型也是一个线性核函数的模型，不
# 同的地方是，它使用了随机梯度下降做训练，所以每次并没有使用全部的样本，收敛速度会快很
# 多。再多提一点，SGDClassifier对于特征的幅度非常敏感，也就是说，我们在把数据灌给它之前，
# 应该先对特征做幅度调整，当然，用sklearn的StandardScaler可以很方便地完成这一点。
# SGDClassifier每次只使用一部分(mini-batch)做训练，在这种情况下，我们使用交叉验
# 证(cross-validation)并不是很合适，我们会使用相对应的progressive validation：
# 简单解释一下，estimator每次只会拿下一个待训练batch在本次做评估，然后训练完之后，
# 再在这个batch上做一次评估，看看是否有优化。

import  numpy as np
from sklearn.datasets import make_classification
import  matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

X, y = make_classification(200000,
                           n_features=200,
                           n_informative=25,
                           n_redundant=0,
                           n_classes=10,
                           class_sep=2,
                           random_state=0)

# 用SGDClassifier做训练，并画出batch在训练前后的得分差
from sklearn.linear_model import SGDClassifier

est = SGDClassifier(penalty="l2",
                    alpha=0.001)
progressive_validation_score = []
train_score = []
for datapoint in range(0, 199000, 1000):
    X_batch = X[datapoint:datapoint + 1000]
    y_batch = y[datapoint:datapoint + 1000]
    if datapoint > 0:
        progressive_validation_score.append(est.score(X_batch, y_batch))
    est.partial_fit(X_batch, y_batch, classes=range(10))
    if datapoint > 0:
        train_score.append(est.score(X_batch, y_batch))

plt.plot(train_score, label="train score")
plt.plot(progressive_validation_score, label="progressive validation score")
plt.xlabel("Mini-batch")
plt.ylabel("Score")
plt.legend(loc='best')
plt.show()

# 从这个图上的得分，我们可以看出在50个mini-batch迭代之后，数据上的得分就已经变化不大了。但是好像得分都不
# 太高，所以我们猜测一下，这个时候我们的数据，处于欠拟合状态。我们刚才在小样本集合上提到了，如果欠拟合，我
# 们可以使用更复杂的模型，比如把核函数设置为非线性的，但遗憾的是像rbf核函数是没有办法和SGDClassifier兼
# 容的。因此我们只能想别的办法了，比如这里，我们可以把SGDClassifier整个替换掉了，用多层感知神经网来完成这
# 个任务，我们之所以会想到多层感知神经网，是因为它也是一个用随机梯度下降训练的算法，同时也是一个非线性的
# 模型。当然根据机器学习算法使用图谱，也可以使用**核估计(kernel-approximation)**来完成这个事情。
