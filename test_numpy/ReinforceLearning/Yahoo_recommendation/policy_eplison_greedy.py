import numpy as np

d = 6
alpha = 2
r1 = 0.6
r2 = -16
episilon = 0.05

class EplisionGreedy():
    def __init__(self):
        self.article_id_to_index = None
        self.acc_reward = None
        self.count = None
        self.max_arm_index = 0
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
        self.article_list = []

        i = 0
        for key in article_id_to_embed_map:
            self.article_id_to_index[key] = i
            self.article_list.append(key)
            i+=1

    def update(self, reward):
        #if reward !=0 and reward != 1: return
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
        self.acc_reward_rate[index] = (self.acc_reward_rate[index]*self.count[index] + r)*1.0/(self.count[index]+1)
        self.acc_reward[index] += r
        self.count[index] +=1

    def recommend(self, time, user_features, candidate_articles):
        #global max_a
        #global x

        #TODO:注意：此处只用user有feature,而arm并没有feature
        #valid_article_indexs = np.array([self.article_id_to_index[article] for article in candidate_articles]) # list
        rand = np.random.uniform()
        # exploration
        if rand <= episilon:
            rand_index = np.random.randint(self.count.shape[0])
            #self.max_a_index = candidate_articles[rand_index]
            self.max_arm_index = rand_index
            return self.article_list[rand_index]
        else:
            #expolitation
            #valid_articles = self.acc_reward_rate[valid_irticle_indexs]
            max_index = self.acc_reward_rate.argmax()
            self.max_arm_index = max_index
            return self.article_list[max_index]
