# blog:https://www.cnblogs.com/yudanqu/p/9031869.html
"""
Seaborn是对matplotlib的extend，是一个数据可视化库，提供更高级的API封装，在应用中更加的方便灵活。下面我简单介绍一下他的用法，实际应用的时候，可以直接从文档中查找这个库，这时候使用就很快捷了。

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
import scipy.cluster
import seaborn as sns


"""
　如果在交互式环境中，可以通过%matplotlab来解决每次通过plt.show()来显示图像（本人在ipython中，貌似jupyter中%matplotlib inline等等）
"""


"""
直方图和密度图
"""
s1 = Series(np.random.randn(1000)) # 生成1000个点的符合正态分布的随机数
plt.hist(s1) # 直方图，也可以通过plot(),修改里面kind参数实现
s1.plot(kind='kde') # 密度图
plt.show()




sns.distplot(s1,hist=True,kde=True,rug=True) # 前两个默认就是True,rug是在最下方显示出频率情况，默认为False
# bins=20 表示等分为20份的效果，同样有label等等参数
sns.kdeplot(s1,shade=True,color='r') # shade表示线下颜色为阴影,color表示颜色是红色
sns.rugplot(s1) # 在下方画出频率情况
plt.show()


"""
柱状图和热力图
"""

df = sns.load_dataset('flights') # 在线下载一个数据用于实验，在sns.load_dataset()函数里有很多的数据，想了解更多的可以到GitHub中找到源码，你就会很清楚了
df = df.pivot(index='month',columns='year',values='passengers') # 生成一个透视表，得到一个以年、月为轴的二维数据表
print(df)
