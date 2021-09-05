import numpy as np
from sklearn import preprocessing

input_data = np.array([
    [5.1, -2.9, 3.3],
    [-1.2, 7.8, -1.6],
    [3.9, 0.4, 2.1],
    [7.3, -9.9, -4.5]
])

# Binarization  二值化
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print('Bianrized data is \n', data_binarized)

# Mean removal  均值移除

data_scaled = preprocessing.scale(input_data)
print('after scaled, mean \n', data_scaled)
print('after scaled, mean = \n', data_scaled.mean(axis=0))
print('std deviation = \n', data_scaled.std(axis=0))

# Min max scaling 线性归一化
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("Min max scaled \n", data_scaled_minmax)

# Normalization 归一化
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("L1 normalized \n", data_normalized_l1)
print("L2 normalized \n", data_normalized_l2)


