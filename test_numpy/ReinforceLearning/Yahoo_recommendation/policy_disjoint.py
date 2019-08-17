import numpy as np

d = 6
alpha = 5
r1 = 0.8
r2 = -20
# yahoo官方 linucb代码

class DisjointPolicy():

    def __init__(self):
        self.article_id_to_index = None
        self.Aa = None
        self.Aa_inv = None
        self.ba = None
        self.theta = None
        self.max_arm_index = 0
        self.x =None
        print("class name:"+self.__class__.__name__)

    def set_articles(self, article_id_to_embed_map):
        """
         TODO: 在Disjoint Linear Models中，article 的embeding并没有使用
        """

        # articles: id -> embedding map
        n_articles = len(article_id_to_embed_map) # 文章个数为k, 即k个arm
        self.article_id_to_index = {}
        self.Aa = np.zeros((n_articles, d, d)) # [k, d, d], k个arm, 每个arm d维
        self.Aa_inv = np.zeros((n_articles, d, d)) # [k, d, d], k个arm, 每个arm d维
        self.ba = np.zeros((n_articles, d, 1)) # [k, d, 1], k个arm,每个arm d维的pay-off向量
        self.theta = np.zeros((n_articles, d, 1)) # [k, d, 1] , Aa*theta = ba, k个arm,每个arm 有d维的参数theta
        i = 0
        for key in article_id_to_embed_map:
            self.article_id_to_index[key] = i
            self.Aa[i] = np.identity(d) # 每个 article有自己的Aa
            self.Aa_inv[i] = np.identity(d)
            self.ba[i] = np.zeros((d, 1))
            self.theta[i] = np.zeros((d, 1))
            i += 1

    def update(self, reward):
        if reward == 1 or reward == 0:
            if reward == 1:
                r = r1
            else:
                r = r2
            #print("max_a:", self.max_a, "Aa:", self.Aa," Aa[max_a]:", self.Aa[self.max_a])

            """
            用reward更新选中arm的参数
            """
            # x:[d, 1],其中x:为user_feature向量
            # 论文中算法1:Aa <- Aa + x_(t,a)*x_(t,a)^T
            self.Aa[self.max_arm_index] += np.outer(self.x, self.x) # Aa:[d, d]
            self.Aa_inv[self.max_arm_index] = np.linalg.inv(self.Aa[self.max_arm_index]) # Aa_inv:[d, d]
            # 更新arm的收益ba
            # 论文中算法1:ba <- ba + r_t*x_(t,a)
            self.ba[self.max_arm_index] += r * self.x
            # 更新theta,论文中算法1: Aa*theta = ba  => theta_a <- A_a^(-1)*ba
            self.theta[self.max_arm_index] = self.Aa_inv[self.max_arm_index].dot(self.ba[self.max_arm_index])
        else:
            # 未命中arm,此次事件将被完全忽略
            pass

    def recommend(self, time, user_features, candidate_articles):
        #global max_a
        #global x

        # TODO:注意：在disjoint中只用user有feature,而arm feature并没有用
        article_len = len(candidate_articles)
        # user_feature为xt,即下面的文章序列均为同一个用户的行为序列
        self.x = np.array(user_features).reshape((d,1)) # x:[d, 1]
        x_t = np.transpose(self.x) # x_t:[1, d]
        candidate_article_indexs = [self.article_id_to_index[article] for article in candidate_articles] # list

        # 取目标arm的前k个arm序列
        # theta: [k, d, 1]
        # x:[d, 1]
        article_thetas = self.theta[candidate_article_indexs] # [k, d, 1]
        article_thetas_trans = np.transpose(article_thetas, (0, 2, 1)) # [k, 1, d]
        # 论文中算法1:p_{t,a}=theta*x_{t,a} + alpha*sqrt(x_{t,a}^T*A_a*x_{t,a})
        exploitation = np.matmul(article_thetas_trans, self.x) # [k, 1, 1]
        # xt:[1, d], Aa_inv:[k, d, d], x:[d, 1]
        A_inv_x = self.Aa_inv[candidate_article_indexs].dot(self.x) # [k, d, 1]
        exploration = np.sqrt(np.matmul(x_t, A_inv_x)) # [k, 1, 1]

        # 所有的文章都算一次ucb，而只需要选出收益最大的即可
        reward = exploitation + alpha * exploration

        max_index = np.argmax(reward)
        self.max_arm_index = candidate_article_indexs[max_index]
        return candidate_articles[max_index]
