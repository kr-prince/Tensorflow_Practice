
# imports
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Read data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Setting up Variables
image_size = 28
labels_size = 10
learning_rate = 0.05
steps_number = 5000
batch_size = 100

# Defining placeholders for training data and respective labels
training_data = tf.placeholder(tf.float32, [None, image_size*image_size])
labels = tf.placeholder(tf.float32, [None, labels_size])

## Variables and weights to be tuned
# weights are assigned with normal distribution
W = tf.Variable(tf.truncated_normal([image_size*image_size, labels_size], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[labels_size]))

# Build the network (only output layer)
output = tf.matmul(training_data, W) + b


# NOTE : TensorFlow provides the function called tf.nn.softmax_cross_entropy_with_logits 
# that internally applies the softmax on the model's unnormalized prediction 
# and sums across all classes. The tf.reduce_mean function takes the average over these sums. 
# This way we get the function that can be further optimised. In our example, we 
# use gradient descent method from the tf.train API

# Defining loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output))

# Training step
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Prediction and accuracy
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Now we completed building the computational graph
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


for i in range(steps_number):
    # Get the next batch
    input_batch, labels_batch = mnist.train.next_batch(batch_size)
    feed_dict = {training_data : input_batch, labels : labels_batch}
    
    # Run the training step
    train_step.run(feed_dict = feed_dict)
    
    # Print the accuracy every 100 steps
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict=feed_dict)
        print("Step %d, training batch accuracy %g %%" %(i, train_accuracy*100))


# Evaluate on the test set
test_accuracy = accuracy.eval(feed_dict={training_data: mnist.test.images, labels: mnist.test.labels})
print("Test accuracy : %g %%" %(test_accuracy*100))
