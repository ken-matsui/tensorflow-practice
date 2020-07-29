# coding:utf-8

import numpy as np
import tensorflow as tf
# mnist-data
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

tf.set_random_seed(0)

X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
Y = np.array([[0, 0], [1, 1], [1, 0], [0, 1], [0, 0], [0, 1], [1, 0], [1, 1]])

'''
Input(3)-Hidden(5)-Hidden(5)-Hidden(5)-Hidden(5)-Output(2)
'''
# 入力層の次元
x = tf.placeholder(tf.float32, shape = [None, 3])
# 出力層の次元
t = tf.placeholder(tf.float32, shape = [None, 2])

# 入力層 - 隠れ層
# weight 3 * 5 matrix
W1 = tf.Variable(tf.truncated_normal([3, 5]))
b1 = tf.Variable(tf.zeros([5]))
# 上記二つの変数は下で使用
h1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

# 隠れ層 - 隠れ層
W2 = tf.Variable(tf.truncated_normal([5, 5]))
b2 = tf.Variable(tf.zeros([5])) # h1は前の層
h2 = tf.nn.sigmoid(tf.matmul(h1, W2) + b2)

W3 = tf.Variable(tf.truncated_normal([5, 5]))
b3 = tf.Variable(tf.zeros([5]))
h3 = tf.nn.sigmoid(tf.matmul(h2, W3) + b3)

W4 = tf.Variable(tf.truncated_normal([5, 3]))
b4 = tf.Variable(tf.zeros([3]))
h4 = tf.nn.sigmoid(tf.matmul(h3, W4) + b4)

# 隠れ層 - 出力層
W5 = tf.Variable(tf.truncated_normal([3, 2]))
b5 = tf.Variable(tf.zeros([2]))
# 上記二つの変数は下で使用
y = tf.nn.sigmoid(tf.matmul(h4, W5) + b5)

cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

# セッションの開始
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# tqdm is progressbar
for epoch in tqdm(range(30000)):
    # 勾配降下法による学習
    # feed_dictと書くことでplaceholderである，x及びtに実際の値を代入している
    # placeholderにfeed(与える) データXを一度に与えている
    sess.run(train_step, feed_dict={
        x: X,
        t: Y
    })

# 学習結果の表示
# 適切に分類できるようになったかどうかの確認 eval()で
classified = correct_prediction.eval(session=sess, feed_dict={
    x: X,
    t: Y
})
prob = y.eval(session=sess, feed_dict={
    x: X
})
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# mnist
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print('\nclassified:')
print(classified)
print()
print('output probability:')
print(prob)
print()
# 予測精度
#print('accuracy:')
#print(accuracy)
#print(mnist)
