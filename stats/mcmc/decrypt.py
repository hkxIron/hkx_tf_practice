# -*- coding: utf-8 -*-
import math
import random

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 根据传入的密钥，生成字母替换规则字典
# 例如传入"DGHJKL..."，生成字典{D:A, G:B, H:C...}
def create_cipher_dict(cipher):
    cipher_dict = {}
    alphabet_list = list(alphabet)
    for i in range(len(cipher)):
        cipher_dict[alphabet_list[i]] = cipher[i]
    return cipher_dict


# 使用密钥对文本进行替换(加密/解密)
def apply_cipher_on_text(text, cipher):
    cipher_dict = create_cipher_dict(cipher)
    text = list(text)
    newtext = ""
    for elem in text:
        if elem.upper() in cipher_dict:
            newtext += cipher_dict[elem.upper()]
        else:
            newtext += " "
    return newtext


# 统计参考语料的bigram
# 例如 {'AB':234,'TH':2343,'CD':23 ..}
def create_scoring_params_dict(longtext_path):
    scoring_params = {}
    alphabet_list = list(alphabet)
    with open(longtext_path) as fp:
        for line in fp:
            data = list(line.strip())
            for i in range(len(data) - 1):
                alpha_i = data[i].upper()
                alpha_j = data[i + 1].upper()
                if alpha_i not in alphabet_list and alpha_i != " ":
                    alpha_i = " "
                if alpha_j not in alphabet_list and alpha_j != " ":
                    alpha_j = " "
                key = alpha_i + alpha_j
                if key in scoring_params:
                    scoring_params[key] += 1
                else:
                    scoring_params[key] = 1
    return scoring_params


# 统计解密文本的bigram
# 例如 {'AB':234,'TH':2343,'CD':23 ..}
def score_params_on_cipher(text):
    scoring_params = {}
    alphabet_list = list(alphabet)
    data = list(text.strip())
    for i in range(len(data) - 1):
        alpha_i = data[i].upper()
        alpha_j = data[i + 1].upper()
        if alpha_i not in alphabet_list and alpha_i != " ":
            alpha_i = " "
        if alpha_j not in alphabet_list and alpha_j != " ":
            alpha_j = " "
        key = alpha_i + alpha_j
        if key in scoring_params:
            scoring_params[key] += 1
        else:
            scoring_params[key] = 1
    return scoring_params


# 根据公式计算密钥的评分
def get_cipher_score(text, cipher, scoring_params):
    # 将密文用当前密钥解密
    decrypted_text = apply_cipher_on_text(text, cipher)
    # 当前密钥解出的密文的 k-count字典
    # 它的分布与参考文章越相近，分越高
    scored_f = score_params_on_cipher(decrypted_text)
    cipher_score = 0
    #for k, v in scored_f.iteritems():
    # 如果他们有共同的key，那么计算共同的value的指数值
    for k, v in scored_f.items():
        if k in scoring_params:
            cipher_score += v * math.log(scoring_params[k]) # y log p ,有点像交叉熵，不过这里是频数, 其中y与p来自不同的分布，一个是参考样本的，一个是密文的
    return cipher_score


# 通过随机交换两个字母的顺序 生成一个新的密钥
def generate_cipher(cipher):
    pos1 = random.randint(0, len(list(cipher)) - 1)
    pos2 = random.randint(0, len(list(cipher)) - 1)
    if pos1 == pos2:
        return generate_cipher(cipher)
    else:
        cipher = list(cipher)
        pos1_alpha = cipher[pos1]
        pos2_alpha = cipher[pos2]
        cipher[pos1] = pos2_alpha
        cipher[pos2] = pos1_alpha
        return "".join(cipher)


# 抛一枚出现正面概率为p的硬币，出现正面返回True，出现反面返回False
# 以概率p接受转移
def random_coin(p):
    unif = random.uniform(0, 1)
    if unif >= p:
        return False
    else:
        return True


# MCMC方法解密 运行n_iter轮
def MCMC_decrypt(n_iter, cipher_text, scoring_params):
    current_cipher = alphabet  # 以随机密钥开始
    state_keeper = set()
    best_state = ''
    score = 0
    for i in range(n_iter):
        state_keeper.add(current_cipher)
        proposed_cipher = generate_cipher(current_cipher)
        # 评估当前密钥
        score_current_cipher = get_cipher_score(cipher_text, current_cipher, scoring_params)
        score_proposed_cipher = get_cipher_score(cipher_text, proposed_cipher, scoring_params)
        score_diff = score_proposed_cipher - score_current_cipher
        if i==10: print("score_current_cipher:",score_current_cipher," score_proposed_cipher:",score_proposed_cipher) # 4918
        # 如果新分比较高，肯定要转移，但如果分较低，以一定概率转移，exp(x-y)=exp(x)/exp(y)
        # 否则以概率ScoreP/ScoreC进行转移
        acceptance_probability = min(1, math.exp(score_diff))
        # if score_current_cipher > score:
        #     best_state = current_cipher
        if random_coin(acceptance_probability):
            print("Trans!iter:",i," p:",acceptance_probability," score:",score_current_cipher,"score_diff:",score_diff)
            current_cipher = proposed_cipher
            #score =  score_proposed_cipher
        if i % 500 == 0:
            print("iter", i," score:",score_current_cipher,"score_diff:",score_diff, " accpet_p:",acceptance_probability," ->", apply_cipher_on_text(cipher_text, current_cipher)[0:99])
    return state_keeper, current_cipher


# 主程序开始

# 参考语料：《战争与和平》
# scoring_params:返回统计后的词典，key -> count
scoring_params = create_scoring_params_dict('war_and_peace_new.txt')
# 测试文本
plain_text = "As Oliver gave this first proof of the free and proper action of his lungs, \
the patchwork coverlet which was carelessly flung over the iron bedstead, rustled; \
the pale face of a young woman was raised feebly from the pillow; and a faint voice imperfectly \
articulated the words, Let me see the child, and die. \
The surgeon had been sitting with his face turned towards the fire: giving the palms of his hands a warm \
and a rub alternately. As the young woman spoke, he rose, and advancing to the bed's head, said, with more kindness \
than might have been expected of him: "

# 来源于 A Singularly Valuable Decomposition: The SVD of a Matrix
# 发现可以解析出来
plain_text2="""
Every teacher of linear algebra should be familiar with the matrix singular value decomposition.
It has interesting and attractive algebraic properties, and conveys important geometrical and
theoretical insights about linear transformations. The close connection between the singular value decomposition and the well
known theory of diagonalization for symmetric matrices makes the topic immediately accessible to linear
algebra teachers, and indeed, a natural extension of what these teachers already know. At the same
time, the singular value decomposition has fundamental importance in several different applications of linear algebra. Strang
was aware of these facts when he introduced the singular value decomposition in his now classical text.
"""

# 缩短文本长度,发现不能解析出文本，因为没有统计意义
plain_text3="""
Every teacher of linear algebra should be familiar with the matrix singular value decomposition.
"""

# 使用密钥对文本进行加密
encryption_key = "XEBPROHYAUFTIDSJLKZMWVNGQC"
cipher_text = apply_cipher_on_text(plain_text2, encryption_key)
#cipher_text = apply_cipher_on_text(plain_text2, encryption_key)
#cipher_text = apply_cipher_on_text(plain_text, encryption_key)
decryption_key = "ICZNBKXGMPRQTWFDYEOLJVUAHS"

print("Text To Decode:", cipher_text,"\n")
states, best_cipher_state = MCMC_decrypt(20000, cipher_text, scoring_params)
print("Decoded Text:", apply_cipher_on_text(cipher_text, best_cipher_state).lower(),"\n") # 测试发现即使用的是svd的测试文本，也可以解析出有意义的文本,非常厉害!!
print("MCMC KEY FOUND:", best_cipher_state,"\n")
print("ACTUAL DECRYPTION KEY:", decryption_key,"\n")