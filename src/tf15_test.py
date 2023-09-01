import tensorflow  as tf
tf.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)

from tensorflow.examples.tutorials.mnist import input_data
# Load MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Define the model architecture
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = tf.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, padding='same')
        self.pool1 = tf.layers.MaxPooling2D((2, 2), strides=2)
        self.conv2 = tf.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, padding='same')
        self.pool2 = tf.layers.MaxPooling2D((2, 2), strides=2)
        self.flatten = tf.layers.Flatten()
        self.fc1 = tf.layers.Dense(128, activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(10, activation=None)

    def call(self, inputs):
        x = tf.reshape(inputs, shape=[-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)

# Build the model and optimizer
model = Model()
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for _ in range(mnist.train.num_examples // 100):
        batch_x, batch_y = mnist.train.next_batch(100)
        with tf.GradientTape() as tape:
            logits = model(batch_x)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_y))
        gradients = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(gradients, model.variables))

    # Calculate accuracy on validation set
    correct_prediction = tf.equal(tf.argmax(model(mnist.validation.images), 1), tf.argmax(mnist.validation.labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    val_accuracy = accuracy.numpy()
    print("Epoch {}: Validation Accuracy = {:.3f}".format(epoch + 1, val_accuracy))