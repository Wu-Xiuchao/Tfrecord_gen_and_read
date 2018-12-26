import tensorflow as tf 
import numpy as np 


# 导入注释文件
img_info_dir = 'img_info.txt'
with open(img_info_dir,'r') as file:
	lines = file.readlines()
lines = [line.strip().split(' ') for line in lines]


writer = tf.python_io.TFRecordWriter('data.tfrecord')

def gen_single_tfrecord(img,index,writer):
	feature = {
	'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
	'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[index]))
	}
	writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())


sess = tf.Session()
for line in lines:
	path,index = line
	img = tf.read_file(path)
	img = sess.run(img)
	index = int(index)
	gen_single_tfrecord(img,index,writer)
sess.close()
writer.close()

print('finish to write data to tfrecord file!')

