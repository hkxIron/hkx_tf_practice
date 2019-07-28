import numpy as np

d = 6
alpha = 0.01
r1 = 0.6
r2 = -16

class UCB():
    def __init__(self):
        self.article_id_to_index = None
        self.acc_reward = None
        self.count = None
        self.max_arm_index = 0
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
        self.article_list = []

        i = 0
        for key in article_id_to_embed_map:
            self.article_id_to_index[key] = i
            self.article_list.append(key)
            i+=1

    def update(self, reward):
        if reward == 1:
            r = r1
        else:
            r = r2
        #print("max_a:", self.max_a, "Aa:", self.Aa," Aa[max_a]:", self.Aa[self.max_a])

        """
        用reward更新选中arm的参数
        """
        # x:[d, 1]
        index = self.max_arm_index
        self.confidence_bound[index] = alpha* np.log(np.sum(self.count)) / np.sqrt(self.count[index])
        self.acc_reward_rate[index] = (self.acc_reward_rate[index]*self.count[index] + r)*1.0/(self.count[index]+1)
        self.acc_reward[index] += r
        self.count[index] +=1

    def recommend(self, time, user_features, candidate_articles):
        #global max_a
        #global x

        #valid_article_indexs = np.array([self.article_id_to_index[article] for article in candidate_articles]) # list
        # 选择最大的upper bound
        ucb = self.acc_reward_rate + self.confidence_bound
        #print("ucb:", ucb)
        max_index = ucb.argmax()
        self.max_arm_index = max_index
        return self.article_list[max_index]
