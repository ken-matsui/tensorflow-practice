# coding:utf-8

import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(0)
tf.set_random_seed(1)


def inference(x, keep_prob, n_in, n_hiddens, n_out):
    # 重みの生成
    def weight_variable(shape):
        # reluに最適化された重み初期値を使用
        initial = np.sqrt(2.0 / shape[0]) * tf.truncated_normal(shape)
        #initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    # バイアスの生成
    def bias_variable(shape):
        # 全てゼロ
        initial = tf.zeros(shape)
        return tf.Variable(initial)

    # ミニバッチ正規化の適用
    def batch_normalization(shape, x):
        eps = 1e-8
        beta = tf.Variable(tf.zeros(shape))
        gamma = tf.Variable(tf.ones(shape))
        mean, var = tf.nn.moments(x, [0])
        return gamma * (x - mean) / tf.sqrt(var + eps) + beta

    # 入力層 - 隠れ層、隠れ層 - 隠れ層
    # enumerateを使うとiにはインデックスn_hiddenには要素が入る
    for i, n_hidden in enumerate(n_hiddens):
        if i == 0:
            input = x
            input_dim = n_in
        else:
            input = output
            input_dim = n_hiddens[i-1]

        W = weight_variable([input_dim, n_hidden])
        # batch_normalizationをしているためバイアスは不必要
        #b = bias_variable([n_hidden])
        # reluの使用とDropOutの使用
        #h = tf.nn.relu(tf.matmul(input, W) + b)
        #output = tf.nn.dropout(h, keep_prob)
        # batch_normalizationの適用
        u = tf.matmul(input, W)
        h = batch_normalization([n_hidden], u)
        output = tf.nn.relu(h)

    # 隠れ層 - 出力層
    W_out = weight_variable([n_hiddens[-1], n_out])
    b_out = bias_variable([n_out])
    y = tf.nn.softmax(tf.matmul(output, W_out) + b_out)
    return y


def loss(y, t):
    cross_entropy = \
        tf.reduce_mean(-tf.reduce_sum(
                       t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)),
                       reduction_indices=[1]))
    return cross_entropy


def training(loss):
    # 一定の学習率を使用
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    # モメンタム項の適用
    #optimizer = tf.train.MomentumOptimizer(0.01, 0.9)
    # Adamの適用
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.999)

    train_step = optimizer.minimize(loss)
    return train_step


# 精度
def accuracy(y, t):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


if __name__ != '__main__':

    # mnistデータを読み込む(fetch)
    mnist = datasets.fetch_mldata('MNIST original', data_home='.')

    # 大きさ
    n = len(mnist.data)
    N = 30000  # MNISTの一部を使う
    N_train = 20000
    N_validation = 4000
    # mnistの大きさ(N)個からn個の選択
    # つまりrange(n)からランダムに並び替えた数が配列として配列の添字用として生成
    # permutationとshuffleの違いは，permutationはコピーを生成するが，
    #                                shuffleはそれ自身を並び替える
    indices = np.random.permutation(range(n))[:N]  # ランダムにN枚を選択

    # ランダムに選択された配列をそのまま添字に与えるとその中身を全て与えることができる
    # mnist.data[[3,1,4,2,5]] = [mnist.data[3],mnist.data[1],......]
    X = mnist.data[indices]
    y = mnist.target[indices]
    # np.eye(10)は10*10の単位行列の生成
    # TODO:その後の[y.astype(int)]はどゆ意味?
    Y = np.eye(10)[y.astype(int)]  # 1-of-K 表現に変換

    # MNISTDATAを訓練データとテストデータに分割
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=N_train)

    # 訓練データをさらに訓練データと検証データに分割
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=N_validation)

    '''
    モデル設定
    '''
    n_in = len(X[0])
    n_hiddens = [200, 200, 200]  # 各隠れ層の次元数
    n_out = len(Y[0])
    p_keep = 0.5 # dropoutしないので本当は不必要

    x = tf.placeholder(tf.float32, shape=[None, n_in])
    t = tf.placeholder(tf.float32, shape=[None, n_out])
    keep_prob = tf.placeholder(tf.float32)

    # キーワード指定で引数に値を渡す．つまり，引数の順番を考慮する必要がなくなる．
    y = inference(x, keep_prob, n_in=n_in, n_hiddens=n_hiddens, n_out=n_out)
    loss = loss(y, t)
    train_step = training(loss)

    accuracy = accuracy(y, t)

    # 辞書の生成
    history = {
        'val_loss': [],
        'val_acc': []
    }

    '''
    モデル学習
    '''
    epochs = 100
    # epochs = 50
    # ミニバッチそれぞれのサイズ
    batch_size = 200

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # 訓練データをミニバッチのサイズごとに分割(切り捨て除算)
    n_batches = N_train // batch_size

    # N個(n_batches)の全データに対する反復回数
    for epoch in tqdm(range(epochs)):
        # エポック毎にデータをシャッフルしてやると偏りが生じにくくなる
        X_, Y_ = shuffle(X_train, Y_train)

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            sess.run(train_step, feed_dict={
                x: X_[start:end],
                t: Y_[start:end],
                keep_prob: p_keep
            })

        # 検証データを用いた評価
        val_loss = loss.eval(session=sess, feed_dict={
            x: X_validation,
            t: Y_validation,
            keep_prob: 1.0
        })
        val_acc = accuracy.eval(session=sess, feed_dict={
            x: X_validation,
            t: Y_validation,
            keep_prob: 1.0
        })

        # 検証データに対する学習の進み具合を記録
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)


    # matplotlibによるグラフの可視化
    # accuracyのplot(追加)
    plt.plot(range(epochs), history['val_acc'], label='accuracy', color='black')
    # lossのplot(追加)
    plt.plot(range(epochs), history['val_loss'], label='loss', color='gray')
    # y軸の範囲を指定
    plt.ylim(0.0, 1.0)
    # 凡例の表示
    plt.legend(loc='best')
    # グラフタイトル
    plt.title('loss and accuracy')
    # グラフとして出力する
    plt.show()
    # 画像として保存する
    #plt.savefig('mnist_tensorflow.eps')

    '''
    予測精度の出力
    '''
    accuracy_rate = accuracy.eval(session=sess, feed_dict={
        x: X_test,
        t: Y_test,
        keep_prob: 1.0
    })
    print('accuracy: ', accuracy_rate)
