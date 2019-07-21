import numpy as np

d = 6
alpha = 5
r1 = 0.6
r2 = -16

class Policy():

    def __init__(self):
        self.index_all = None
        self.Aa = None
        self.Aa_inv = None
        self.ba = None
        self.theta = None
        self.max_a = 0
        self.x =None

    def set_articles(self, articles):

        n_articles = len(articles)
        self.index_all = {}
        self.Aa = np.zeros((n_articles, d, d))
        self.Aa_inv = np.zeros((n_articles, d, d))
        self.ba = np.zeros((n_articles, d, 1))
        self.theta = np.zeros((n_articles, d, 1))
        i = 0
        for key in articles:
            self.index_all[key] = i
            self.Aa[i] = np.identity(d)
            self.Aa_inv[i] = np.identity(d)
            self.ba[i] = np.zeros((d, 1))
            self.theta[i] = np.zeros((d, 1))
            i += 1
        #pass


    def update(self, reward):
        if reward == -1:
            pass
        elif reward == 1 or reward == 0:
            if reward == 1:
                r = r1
            else:
                r = r2
            #print("max_a:", self.max_a, "Aa:", self.Aa," Aa[max_a]:", self.Aa[self.max_a])
            self.Aa[self.max_a] += np.outer(self.x, self.x)
            self.Aa_inv[self.max_a] = np.linalg.inv(self.Aa[self.max_a])
            self.ba[self.max_a] += r * self.x
            self.theta[self.max_a] = self.Aa_inv[self.max_a].dot(self.ba[self.max_a])
        else:
            pass

    def recommend(self, time, user_features, choices):
        #global max_a
        #global x

        article_len = len(choices)

        self.x = np.array(user_features).reshape((d,1))
        x_t = np.transpose(self.x)
        index = [self.index_all[article] for article in choices]
        UCB = np.matmul(np.transpose(self.theta[index],(0,2,1)), self.x) + alpha * np.sqrt(np.matmul(x_t, self.Aa_inv[index].dot(self.x)))

        max_index = np.argmax(UCB)
        max_a = index[max_index]
        return choices[max_index]
