
dict = {'Name': 'Zara', 'Age': 7,"school":"yale"}
dict2 = {'Sex': 'female' ,'school':"mit"}

dict.update(dict2) # 将dict2的值都赋给dict
print("dict:",dict," dict2:",dict2)
print(dict.get("school",-1))

import multiprocessing
import subprocess
from multiprocessing.pool import ThreadPool
import os
import numpy as np
#os.rmdir()
#os.remove()
data = [ [0,2],[3,4] ]
mat = np.array(data,dtype=np.int64)
print("mat:",mat)

d1=np.array([1,2,3])
d2=np.array([4,5,6])
print(" stack: ",np.stack([d1,d2],axis=1))