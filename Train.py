
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder("float", [None, 10])  

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

y_output = tf.nn.softmax(tf.nn.relu(tf.matmul(fc1_dropout, W_fc2) / dropout1 + b_fc2))  # I think we need compensate here

cross_entropy = -tf.reduce_sum(y*tf.log(y_output))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_output,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

model_path="checkpoint/variable"
saver = tf.train.Saver()

sess.run(tf.initialize_all_variables())

for i in range(50000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y: batch[1], dropout1: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y: batch[1], dropout1: 0.5})
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y: mnist.test.labels, dropout1: 1.0}))

save_path = saver.save(sess, model_path)


