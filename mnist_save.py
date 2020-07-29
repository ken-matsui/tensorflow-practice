# coding: utf-8

# Inside
import os
import sys
import queue
import threading
# Outside
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

def main():
	try:
		os.mkdir('./mnist_images')
	except FileExistsError:
		pass

	que = queue.Queue()
	def saver(que=que):
		while True:
			img, NOW = que.get()
			img.save("./mnist_images/mnist_{}.png".format(NOW))
			que.task_done()

	for _ in range(4):
		t = threading.Thread(target=saver)
		t.setDaemon(True)
		t.start()

	MAX = len(mnist.train.images)
	for i, img in enumerate(mnist.train.images):
		b = img.reshape((28,28))
		b = b * 255
		outImg = Image.fromarray(b)
		outImg = outImg.convert("RGB").resize((96, 96))
		que.put([outImg, i])
		bar, percent = calc_bar(i+1, MAX)
		sys.stdout.write("\r{}/{} [{}] - {}%".format(i+1, MAX, bar, percent))
	que.join()

	print("\ndone.")

def calc_bar(now_count, max_count):
	max_bar_size = 50
	percent = (now_count*100) // max_count
	bar_num = percent // 2
	bar = ""
	if (bar_num - 1) > 0:
		for _ in range(bar_num - 1):
			bar += "="
		bar += ">"
		for _ in range(max_bar_size - bar_num):
			bar += " "
	elif bar_num == 1:
		bar = ">"
		for _ in range(max_bar_size - 1):
			bar += " "
	else:
		for _ in range(max_bar_size):
			bar += " "
	return bar, percent

if __name__ == '__main__':
	main()
