# 3.2 机器学习算法选择
# 数据的情况我们大致看了一眼，确定一些特征维度之后，我们可以考虑先
# 选用机器学习算法做一个baseline的系统出来了。这里我们继续参照上面
# 提到过的机器学习算法使用图谱。
# 我们只有1000个数据样本，是分类问题，同时是一个有监督学习，因此
# 我们根据图谱里教的方法，使用LinearSVC(support vector
# classification with linear kernel)试试。注意，LinearSVC需要选择
# 正则化方法以缓解过拟合问题；我们这里选择使用最多的L2正则化，并把
# 惩罚系数C设为10。我们改写一下sklearn中的学习曲线绘制函数，画出
# 训练集和交叉验证集上的得分
# ---------------------


import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import learning_curve
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif



X, y = make_classification(1000,
                           n_features=20,
                           n_informative=2,
                           n_redundant=2,
                           n_classes=2,
                           random_state=0)


# 绘制学习曲线，以确定模型的状况
def plot_learning_curve(estimator,
                        title,
                        X,
                        y,
                        ylm,
                        cv,
                        train_sizes):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    """


    train_sizes, train_scores, test_scores = learning_curve(estimator,
                                                            X,
                                                            y,
                                                            cv = cv,
                                                            n_jobs = 1,
                                                            train_sizes = train_sizes)
    train_scores_mean = np.mean(train_scores,
                                axis=1)
    train_scores_std = np.std(train_scores,
                              axis=1)
    test_scores_mean = np.mean(test_scores,
                               axis=1)
    test_scores_std = np.std(test_scores,
                             axis=1)

    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color="g")
    plt.plot(train_sizes,
             train_scores_mean,
             'o-', color="r",
             label="Training score")
    plt.plot(train_sizes,
             test_scores_mean,
             'o-', color="g",
             label="Cross-validation score")

    plt.xlabel("Training Samples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on")
    if ylm:
        plt.ylim(ylm)
    plt.title(title)
    # plt.show()


# 少样本的情况情况下绘出学习曲线
plt.figure()
plot_learning_curve(LinearSVC(C=10.0),
                    "LinearSVC(C=10.0)",
                    X,
                    y,
                    ylm = (0.8, 1.1),
                    cv = 5,
                    train_sizes = np.linspace(.05, 0.2, 5))


# 这幅图上，我们发现随着样本量的增加，训练集上的得分有一定程度的下降，
# 交叉验证集上的得分有一定程度的上升，但总体说来，两者之间有很大的差距，
# 训练集上的准确度远高于交叉验证集。这其实意味着我们的模型处于过拟合的状态，
# 也即模型太努力地刻画训练集，一不小心把很多噪声的分布也拟合上了，
# 导致在新数据上的泛化能力变差了
# ---------------------

# 3.2.1 过拟合的定位与解决
# 问题来了，过拟合咋办？
# 针对过拟合，有几种办法可以处理：
# （一）增大样本量
# 这个比较好理解吧，过拟合的主要原因是模型太努力地去记住训练样本的分布状况，
# 而加大样本量，可以使得训练集的分布更加具备普适性，噪声对整体的影响下降。
# 恩，我们提高点样本量试试：

plt.figure()
plot_learning_curve(LinearSVC(C=10.0),
                    "LinearSVC(C=10.0)",
                    X,
                    y,
                    ylm = (0.8, 1.1),
                    cv = 5,
                    train_sizes = np.linspace(.1, 1, 5))

# 是不是发现问题好了很多？随着我们增大训练样本量，我们发现训练集和交叉验证集上
# 的得分差距在减少，最后它们已经非常接近了。增大样本量，最直接的方法当然是想办
# 法去采集相同场景下的新数据，如果实在做不到，也可以试试在已有数据的基础上做一
# 些人工的处理生成新数据(比如图像识别中，我们可能可以对图片做镜像变换、旋转等
# 等)，当然，这样做一定要谨慎，强烈建议想办法采集真实数据。


# （二）减少特征的量(只用我们觉得有效的特征)
# 比如在这个例子中，我们之前的数据可视化和分析的结果表明，第11和14维特征包含的
# 信息对识别类别非常有用，我们可以只用它们。
plt.figure()
plot_learning_curve(LinearSVC(C=10.0),
                    "LinearSVC(C=10.0) Features: 11&14",
                    X[:, [11, 14]],
                    y,
                    ylm=(0.8, 1.1),
                    cv = 5,
                    train_sizes = np.linspace(.05, 0.2, 5))

# 从上图上可以看出，过拟合问题也得到一定程度的缓解。不过我们这是自己观察后，手
# 动选出11和14维特征。那能不能自动进行特征组合和选择呢，其实我们当然可以遍历特
# 征的组合样式，然后再进行特征选择(前提依旧是这里特征的维度不高，如果高的话，遍
# 历所有的组合是一个非常非常非常耗时的过程！！)：
plt.figure()
plot_learning_curve(Pipeline([("fs", SelectKBest(f_classif, k=2)), # select two features
                               ("svc", LinearSVC(C=10.0))]),
                    "SelectKBest(f_classif, k=2) + LinearSVC(C=10.0)",
                    X,
                    y,
                    ylm=(0.8, 1.1),
                    cv = 5,
                    train_sizes=np.linspace(.05, 0.2, 5))

# 如果你自己跑一下程序，会发现在我们自己手造的这份数据集上，这个特征筛选的过程
# 超级顺利，但依旧像我们之前提过的一样，这是因为特征的维度不太高。
# 从另外一个角度看，我们之所以做特征选择，是想降低模型的复杂度，而更不容易刻画
# 到噪声数据的分布。从这个角度出发，我们还可以有(1)模型中降低多项式次数 (2)神经
# 网络中减少神经网络的层数和每层的结点数 ©SVM中增加RBF-kernel的bandwidth等方式
# 来降低模型的复杂度。
# 话说回来，即使以上提到的办法降低模型复杂度后，好像能在一定程度上缓解过拟合，
# 但是我们一般还是不建议一遇到过拟合，就用这些方法处理，优先用下面的方法：
# （三）增强正则化作用(比如说这里是减小LinearSVC中的C参数)
# 正则化是我认为在不损失信息的情况下，最有效的缓解过拟合现象的方法。
plt.figure()
plot_learning_curve(LinearSVC(C=0.1),
                    "LinearSVC(C=0.1)",
                    X,
                    y,
                    ylm=(0.8, 1.1),
                    cv = 5,
                    train_sizes=np.linspace(.05, 0.2, 5))

# 调整正则化系数后，发现确实过拟合现象有一定程度的缓解，但依旧是那个问题，我们现在
# 的系数是自己敲定的，有没有办法可以自动选择最佳的这个参数呢？可以。我们可以在交叉
# 验证集上做grid-search查找最好的正则化系数(对于大数据样本，我们依旧需要考虑时间问
# 题，这个过程可能会比较慢):
from sklearn.model_selection import GridSearchCV
estm = GridSearchCV(LinearSVC(),
                   param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10.0]})
plt.figure()
plot_learning_curve(estm,
                    "LinearSVC(C=AUTO)",
                    X,
                    y,
                    ylm=(0.8, 1.1),
                    cv = 5,
                    train_sizes=np.linspace(.05, 0.2, 5))
print("Chosen parameter on 100 datapoints: %s" % estm.fit(X[:500], y[:500]).best_params_)

# 对于特征选择的部分，我打算多说几句，我们刚才看过了用sklearn.feature_selection中的
# SelectKBest来选择特征的过程，也提到了在高维特征的情况下，这个过程可能会非常非常慢。
# 那我们有别的办法可以进行特征选择吗？比如说，我们的分类器自己能否甄别那些特征是对最
# 后的结果有益的？这里有个实际工作中用到的小技巧。
# 我们知道：
#
# l2正则化，它对于最后的特征权重的影响是，尽量打散权重到每个特征维度上，不让权重集中
# 在某些维度上，出现权重特别高的特征。
# 而l1正则化，它对于最后的特征权重的影响是，让特征获得的权重稀疏化，也就是对结果影响
# 不那么大的特征，干脆就拿不着权重。
#
# 那基于这个理论，我们可以把SVC中的正则化替换成l1正则化，让其自动甄别哪些特征应该留
# 下权重。
plt.figure()
plot_learning_curve(LinearSVC(C=0.1, penalty='l1', dual=False),
                    "LinearSVC(C=0.1, penalty='l1')",
                    X,
                    y,
                    ylm=(0.8, 1.1),
                    cv = 5,
                    train_sizes=np.linspace(.05, 0.2, 5))

# 好了，我们一起来看看最后特征获得的权重：
estm = LinearSVC(C=0.1, penalty='l1', dual=False)
estm.fit(X[:450], y[:450])  # 用450个点来训练
print("Coefficients learned: %s" % estm.coef_)
print("Non-zero coefficients: %s" % np.nonzero(estm.coef_)[1])

# 3.2.2 欠拟合定位与解决
# 我们再随机生成一份数据[1000*20]的数据(但是分布和之前有变化)，重新使用LinearSVC来
# 做分类。
from sklearn.datasets import make_circles
from pandas import DataFrame
import seaborn as sns
X, y = make_circles(n_samples=1000, random_state=2)
#绘出学习曲线
plt.figure()
plot_learning_curve(LinearSVC(C=0.25),
                    "LinearSVC(C=0.25)",
                    X,
                    y,
                    ylm=(.1, 1.1),
                    cv = 5,
                    train_sizes=np.linspace(.1, 1, 5))
# 简直烂出翔了有木有，二分类问题，我们做随机猜测，准确率都有0.5，这比随机猜测都高不了
# 多少！！！怎么办？
# 不要盲目动手收集更多资料，或者调整正则化参数。我们从学习曲线上其实可以看出来，训练
# 集上的准确度和交叉验证集上的准确度都很低，这其实就对应了我们说的『欠拟合』状态。别
# 急，我们回到我们的数据，还是可视化看看：
plt.figure()
columns = list(range(2)) + ["class"]
df = DataFrame(np.hstack((X, y[:, None])),
               columns = columns)
sns.pairplot(df,
             vars=[0, 1],
             hue="class",
             size=3.5)

# 你发现什么了，数据根本就没办法线性分割！！！，所以你再找更多的数据，或者调整正则化参数，
# 都是无济于事的！！！
# 那我们又怎么解决欠拟合问题呢？通常有下面一些方法：
# （一）调整你的特征(找更有效的特征！！)
# 比如说我们观察完现在的数据分布，然后我们先对数据做个映射：
X_extra = np.hstack((X, X[:, [0]]**2 + X[:, [1]]**2))
plt.figure()
plot_learning_curve(LinearSVC(C=0.25),
                    "LinearSVC(C=0.25) + distance feature",
                    X_extra,
                    y,
                    ylm=(0.5, 1.1),
                    cv = 5,
                    train_sizes=np.linspace(.1, 1.0, 5))
# 卧槽，少年，这准确率，被吓尿了有木有啊！！！所以你看，选用的特征影响太大了，当然，我们
# 这里是人工模拟出来的数据，分布太明显了，实际数据上，会比这个麻烦一些，但是在特征上面下
# 的功夫还是很有回报的。
# （二）使用更复杂一点的模型(比如说用非线性的核函数)
# 我们对模型稍微调整了一下，用了一个复杂一些的非线性rbf kernel：
from sklearn.svm import SVC
# note: we use the original X without the extra feature
plt.figure()
plot_learning_curve(SVC(C=2.5, kernel="rbf", gamma=1.0),
                    "SVC(C=2.5, kernel='rbf', gamma=1.0)",
                    X,
                    y,
                    ylm=(0.5, 1.1),
                    cv = 5,
                    train_sizes=np.linspace(.1, 1.0, 5))
# 你看，效果依旧很赞。
plt.show()

