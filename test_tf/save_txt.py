import numpy as np
import os
file_name="data/arr.txt"
arr= np.random.random((2,3))
print(arr)
#os.rmdir(file_name)
#np.set_printoptions(precision=3)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
arr_str = arr.astype(np.str)
print(arr_str)
np.savetxt(file_name,arr,fmt="%.7f")