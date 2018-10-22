# LDA中平稳分布
# pi(t+1) = pi(t)* A
# A为马氏转移矩阵
# pi(0)为初始状态
# 若A正定,则有 A = P^-1* D * P,因此 A^n = p^-1*(D^n)*p,即感觉并不会形成平稳链, 此认识错误 !!!
# 但对于一般的A而言,只有奇异值分解: A = P*D*Q,其中P为m阶的正交矩阵,Q为n阶的正交矩阵,但并不一定会有PQ=I
# 此时A*A = P*D*Q*P*D*Q,其中矩阵并不能相消!
# A*A*A = P*D*Q *P*D*Q *P*D*Q

import numpy as np
pi = np.array([0.3, 0.4, 0.3]) # 即使pi之和不为1,最后也会收敛
A = np.array([[0.9, 0.075, 0.025], # 注意:马氏链中的矩阵A要求每行之和为1
              [0.15, 0.8, 0.05],
              [0.25, 0.25, 0.5]
              ])

from numpy import linalg
U,sigma,VT=linalg.svd(A)  # U与VT并不一定正交
print("U:{}\n sigma:{}\n VT:{}".format(U, sigma, VT))
print("U*U^T:{}".format(np.dot(U, np.transpose(U)))) # 为单位矩阵I
print("VT*VT^T:{}".format(np.dot(VT, np.transpose(VT)))) # 为单位矩阵I
print("U*VT:{}".format(np.dot(U, VT))) # 注意,此处并不为单位矩阵I
print("U*(VT^T):{}".format(np.dot(U, np.transpose(VT)))) # 注意:此处并不为单位矩阵I

pt = pi
only_A = A
only_sigma = sigma
for i in range(80):
    pt = np.dot(pt, A)
    only_A = np.dot(only_A, A)
    only_sigma = only_sigma*sigma
    if i>=70:
        print("iter:{} pt:{}".format(i, pt)) # 到100次时,pt已经收到 [0.625, 0.3125, 0.0625]
        print("iter:{} only_A:{}".format(i, only_A)) # 到100次时,pt已经收到 [0.625, 0.3125, 0.0625]
        print("iter:{} only_sigma:{}".format(i, only_sigma)) # 到100次时,并不收敛,要么无穷,要么为0

"""
 A^n:
 [[ 0.625   0.3125  0.0625] , x0->x0:0.625, x0->x1:0.3125, x0->x2:0.0625
 [ 0.625   0.3125  0.0625] , x1->x0:0.626, x1->x1:0.3125, x1->x2:0.0625
 [ 0.625   0.3125  0.0625]], x2->x0:0.625, x2->x1:0.3125, x2->x2:0.0625
 
 即可以理解为:任意状态到x0状态概率相同,同理到x1,x2的状态概率分别相同.
"""

