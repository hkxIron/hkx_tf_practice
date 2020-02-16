import numpy as np
from sklearn import datasets, linear_model, metrics

matrix_before = np.array([
             [1,2,-2],
             [-3, 8, 1],
             [1,2,-2],
             [9, 3, 1],
             [0,2,-5],
             [1,-2,-2],
             [-3,4,-5]]
             )

BAGID_INDEX = 2

# 先将包排序，然后计算auc
matrix = matrix_before[matrix_before[:, BAGID_INDEX].argsort()]

print("matrix:",matrix)
last_bag_index=0
num = matrix.shape[0]
for index in range(num):
    if matrix[index, BAGID_INDEX] != matrix[last_bag_index, BAGID_INDEX]:
        bagid = matrix[last_bag_index, BAGID_INDEX]
        bagid_matrix = matrix[last_bag_index:index]
        print("bag_id:",bagid)
        print("bag_id_matrix:",bagid_matrix)
        last_bag_index = index

print("last:",last_bag_index )
bagid = matrix[last_bag_index, BAGID_INDEX]
bagid_matrix = matrix[last_bag_index:,:]
print("bag_id:",bagid)
print("bag_id_matrix:",bagid_matrix)
