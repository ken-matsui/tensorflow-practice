# coding:utf-8

import tensorflow as tf

def hello(str):
	return [ tf.constant(str), tf.Session() ]

if __name__ == '__main__':
	hello, sess = hello('Hello, TensorFlow!')
	print sess.run(hello)