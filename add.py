# coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

def main():
	# 定数
	a = tf.constant(5)
	b = tf.constant(3)
	added = tf.add(a, b)

	# 変数
	# c = tf.Variable(9)
	# 後から代入する．メモリだけ確保的な．プレースホルダー
	# c = tf.placeholder(tf.float32)

	print(added)
	with tf.Session() as sess:
		print(sess.run(added))

if __name__ == '__main__':
	main()