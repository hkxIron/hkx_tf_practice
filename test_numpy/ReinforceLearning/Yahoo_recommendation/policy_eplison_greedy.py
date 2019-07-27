import numpy as np

d = 6
alpha = 5
r1 = 0.6
r2 = -16
# yahoo官方 linucb代码

class EplisionGreedy():

    def __init__(self):
        self.article_id_to_index = None
        self.Aa = None
        self.Aa_inv = None
        self.ba = None
        self.theta = None
        self.max_a = 0
        self.x =None

    def set_articles(self, article_id_to_embed_map):
        """
        在Disjoint Linear Models中，article 的embeding并没有使用
        """

        # articles: id -> embedding map
        n_articles = len(article_id_to_embed_map)
        self.article_id_to_index = {}
        self.Aa = np.zeros((n_articles, d, d)) # [k, d, d], k个arm, 每个arm d维
        self.Aa_inv = np.zeros((n_articles, d, d)) # [k, d, d], k个arm, 每个arm d维
        self.ba = np.zeros((n_articles, d, 1)) # [k, d, 1]
        self.theta = np.zeros((n_articles, d, 1)) # [k, d, 1] , Aa*theta = ba
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
            # x:[d, 1]
            self.Aa[self.max_a] += np.outer(self.x, self.x)
            self.Aa_inv[self.max_a] = np.linalg.inv(self.Aa[self.max_a])
            # 更新arm的收益ba
            self.ba[self.max_a] += r * self.x
            # 更瓣theta, Aa*theta = ba
            self.theta[self.max_a] = self.Aa_inv[self.max_a].dot(self.ba[self.max_a])
        else:
            pass

    def recommend(self, time, user_features, candidate_articles):
        #global max_a
        #global x

        #TODO:注意：此处只用user有feature,而arm并没有feature

        article_len = len(candidate_articles)
        # user_feature为xt
        self.x = np.array(user_features).reshape((d,1)) # x:[d, 1]
        x_t = np.transpose(self.x) # x_t:[1, d]
        article_indexs = [self.article_id_to_index[article] for article in candidate_articles] # list

        # 取前k个arm
        # theta: [k, d, 1]
        # x:[d, 1]
        article_thetas = self.theta[article_indexs] # [k, d, 1]
        article_thetas_trans = np.transpose(article_thetas, (0, 2 ,1)) # [k, 1, d]
        exploitation = np.matmul(article_thetas_trans, self.x) # [k, 1, 1]
        # xt:[1, d], Aa_inv:[k, d, d], x:[d, 1]
        A_inv_x = self.Aa_inv[article_indexs].dot(self.x) # [k, d, 1]
        exploration = np.sqrt(np.matmul(x_t, A_inv_x))

        # 所有的文章都算一次ucb，而只需要选出收益最大的即可
        UCB = exploitation + alpha * exploration

        max_index = np.argmax(UCB)
        max_a = article_indexs[max_index]
        return candidate_articles[max_index]
