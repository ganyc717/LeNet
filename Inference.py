import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
from PIL import Image,ImageOps
import numpy as np

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])

def define_variable(shape,name):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1),name)

input_image = tf.reshape(x, [-1, 28, 28, 1])
W_conv1 = define_variable([5, 5, 1, 6], "W_conv1")
b_conv1 = define_variable([6], "b_conv1")


conv1 = tf.nn.conv2d(input_image, W_conv1, strides=[1, 1, 1, 1],padding='SAME') 
relu1 = tf.nn.relu(conv1 + b_conv1)
pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID') 

W_conv2 = define_variable([5, 5, 6, 16], "W_conv2")
b_conv2 = define_variable([16], "b_conv2")

conv2 = tf.nn.conv2d(pool1, W_conv2, strides=[1, 1, 1, 1],padding='VALID') 
relu2 = tf.nn.relu(conv2 + b_conv2) 
pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


W_conv3 = define_variable([5, 5, 16, 120], "W_conv3")
b_conv3 = define_variable([120], "b_conv3")

conv3 = tf.nn.conv2d(pool2, W_conv3, strides=[1, 1, 1, 1],padding='VALID') 
relu3 = tf.nn.relu(conv3 + b_conv3)

flat = tf.reshape(relu3,[-1,120])

W_fc1 = define_variable([120,84], "W_fc1")
b_fc1 = define_variable([84], "b_fc1")

fc1 = tf.nn.relu(tf.matmul(flat, W_fc1) + b_fc1)

dropout1 = tf.placeholder("float")
fc1_dropout = tf.nn.dropout(fc1, dropout1)

W_fc2 = define_variable([84, 10], "W_fc2")
b_fc2 = define_variable([10], "b_fc2")

y_output = tf.nn.softmax(tf.nn.relu(tf.matmul(fc1_dropout, W_fc2) + b_fc2))


model_path="checkpoint/variable"
saver = tf.train.Saver()

load_path = saver.restore(sess, model_path)


def inference(image):
    img = image.convert('L')
    img = img.resize([28,28],Image.ANTIALIAS)
    x_input = np.array(img,dtype="float32")/255
    x_input = np.reshape(x_input,[-1,784])
    output = y_output.eval(feed_dict={x:x_input,dropout1 : 1.0})
    return tf.argmax(output,1).eval()


#output = y_output.eval(feed_dict={x:img,dropout1 : 1.0})
#a =  tf.argmax(output,1)
#print a.eval()
"""
for i in range(10000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y: batch[1], dropout1: 1.0})
        print "step %d, training accuracy %g"%(i, train_accuracy)
    train_step.run(feed_dict={x: batch[0], y: batch[1], dropout1: 0.5})
print "test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y: mnist.test.labels, dropout1: 1.0})
"""
