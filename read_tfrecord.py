import tensorflow as tf
import matplotlib.pyplot as plt

tfrecord_file = 'data.tfrecord'

sess = tf.Session()


plt.figure(figsize=(9,3))
for i,serialized_example in enumerate(tf.python_io.tf_record_iterator(tfrecord_file)):
	features = tf.parse_single_example(
		serialized_example,
		features = {
		'image': tf.FixedLenFeature([], tf.string),
		'index': tf.FixedLenFeature([], tf.int64)
		}
		)
	image = tf.image.decode_jpeg(features['image'], channels=3)
	image,index = sess.run([image,features['index']])
	plt.subplot(1,3,i+1)
	plt.imshow(image)
	plt.xticks([])
	plt.yticks([])
	plt.xlabel(str(index))


sess.close()
plt.show()