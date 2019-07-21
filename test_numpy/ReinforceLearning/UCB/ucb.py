# blog: https://zhuanlan.zhihu.com/p/21404922
import numpy as np
from scipy import linalg

class LinUCB:

    def __init__(self):

        self.alpha = 0.25

        self.r1 = 1  # if worse -> 0.7, 0.8

        self.r0 = 0  # if worse, -19, -21

        # dimension of user features = d

        self.d = 6

        # Aa : collection of matrix to compute disjoint part for each article a, d*d

        self.Aa = {}

        # AaI : store the inverse of all Aa matrix

        self.AaI = {}

        # ba : collection of vectors to compute disjoin part, d*1

        self.ba = {}

        self.a_max = 0

        self.theta = {}

        self.x = None

        self.xT = None

        # linUCB

    def set_articles(self, art):

        # init collection of matrix/vector Aa, Ba, ba

        for key in art:
            self.Aa[key] = np.identity(self.d)

            self.ba[key] = np.zeros((self.d, 1))

            self.AaI[key] = np.identity(self.d)

            self.theta[key] = np.zeros((self.d, 1))

    """

    这里更新参数时没有传入更新哪个arm，因为在上一次recommend的时候缓存了被选的那个arm，所以此处不用传入

    另外，update操作不用阻塞recommend，可以异步执行

    """

    def update(self, reward):

        if reward == 1 or reward == 0:

            if reward == 1:

                r = self.r1

            else:

                r = self.r0

            # Aa = Aa + xt * xt.T
            # ba = ba + rt * xt
            self.Aa[self.a_max] += np.dot(self.x, self.xT) # x:[d, 1], Aa:[d, d]

            self.ba[self.a_max] += r * self.x # ba:[d, 1], x:[d, 1]

            # 求Aa的逆, [d, d]
            self.AaI[self.a_max] = linalg.solve(self.Aa[self.a_max], np.identity(self.d))

            # theta_a = Aa^(-1)*ba , Aa:[d, d], ba:[d,1], theta_a:[d, 1]
            self.theta[self.a_max] = np.dot(self.AaI[self.a_max], self.ba[self.a_max])

        else:
            # error
            pass

    """

    预估每个arm的回报期望及置信区间

    """

    def recommend(self, timestamp, user_features, articles):

        # xaT:[1, d]
        # xa:[d, 1]
        xaT = np.array([user_features])
        xa = np.transpose(xaT)

        art_max = -1

        old_pa = 0

        # 获取在update阶段已经更新过的AaI(求逆结果)
        # AaI_tmp:[k, d, d], 假设有k篇文章
        AaI_tmp = np.array([self.AaI[article] for article in articles])
        # theta_tmp:[k, d,1]
        theta_tmp = np.array([self.theta[article] for article in articles])

        # 计算最大收益
        # pa = theta_a * xa + alpha* sqrt(xa*Aa^(-1)*xa)
        # theta_tmp:[k, d, 1]
        # xaT:[1, d]
        # xa:[d, 1]
        # art_rewards:[k, 1]
        art_rewards = np.dot(xaT, theta_tmp) + self.alpha * np.sqrt(np.dot(np.dot(xaT, AaI_tmp), xa))
        # 记录最优的art
        art_max = articles[np.argmax(art_rewards)]

        # 缓存选择结果，用于update

        self.x = xa

        self.xT = xaT

        # article index with largest UCB
        self.a_max = art_max

        return self.a_max
