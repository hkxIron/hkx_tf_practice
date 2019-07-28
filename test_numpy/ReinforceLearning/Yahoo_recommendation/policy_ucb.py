import numpy as np

d = 6
alpha = 0.001
r1 = 0.6
r2 = -16

class UCB():
    def __init__(self):
        self.article_id_to_index = None
        self.acc_reward = None
        self.count = None
        self.max_a = 0
        self.confidence_bound = None
        print("class name:"+self.__class__.__name__)

    def set_articles(self, article_id_to_embed_map):
        """
        在Disjoint Linear Models中，article 的embeding并没有使用
        """
        # articles: id -> embedding map
        n_articles = len(article_id_to_embed_map)
        self.article_id_to_index = {}

        self.acc_reward = np.zeros((n_articles, 1))
        self.count = np.ones((n_articles, 1))
        self.acc_reward_rate = np.zeros((n_articles, 1))
        self.confidence_bound = alpha/np.ones((n_articles, 1))

        i = 0
        for key in article_id_to_embed_map:
            self.article_id_to_index[key] = i
            i+=1

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
            index = self.article_id_to_index[self.max_a]
            self.confidence_bound[index] = alpha / np.sqrt(self.count[index])
            self.acc_reward_rate[index] = (self.acc_reward_rate[index]*self.count[index] + r)*1.0/(self.count[index]+1)
            self.acc_reward[index] += r
            self.count[index] +=1

    def recommend(self, time, user_features, candidate_articles):
        #global max_a
        #global x

        valid_article_indexs = np.array([self.article_id_to_index[article] for article in candidate_articles]) # list
        # 选择最大的upper bound
        valid_articles = self.acc_reward_rate[valid_article_indexs] + self.confidence_bound[valid_article_indexs]
        max_index = valid_articles.argmax()
        self.max_a = valid_articles[max_index]
        return self.max_a
