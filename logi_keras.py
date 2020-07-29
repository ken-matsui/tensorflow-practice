# coding:utf-8

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

#np.random.seed(0) # 乱数シード
np.random.rand() # 一様乱数 0~1

'''
モデル設定
'''
model = Sequential([
	Dense(input_dim=2, units=1),
	Activation('sigmoid')
])

# model = Sequential()
# model.add(Dense(input_dim=2, units=1))
# model.add(Activation('sigmoid'))

# lr = 学習係数
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

'''
モデル学習
'''
# ORゲート
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

model.fit(X, Y, epochs=2000, batch_size=1)

'''
学習結果の確認
'''
classes = model.predict_classes(X, batch_size=1)
prob = model.predict_proba(X, batch_size=1)

print()
print('classified:')
print(Y == classes)
print()
print('output probability:')
print(prob)