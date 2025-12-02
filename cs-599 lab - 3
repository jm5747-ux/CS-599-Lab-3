import tensorflow as tf
import numpy as np

# Batch Normalization function (copied from previous cell)
def batch_norm(x, gamma, beta, epsilon=1e-5):
    axes = list(range(len(x.shape) - 1))
    batch_mean = tf.reduce_mean(x, axis=axes, keepdims=True)
    batch_variance = tf.reduce_mean(tf.square(x - batch_mean), axis=axes, keepdims=True)
    x_norm = (x - batch_mean) / tf.sqrt(batch_variance + epsilon)
    return gamma * x_norm + beta

# Weight Normalization function (copied from previous cell)
def weight_norm(v, g, axis=None, epsilon=1e-5):
    v_norm = tf.sqrt(tf.reduce_sum(tf.square(v), axis=axis, keepdims=True) + epsilon)
    return (g / v_norm) * v

# Layer Normalization function (copied from previous cell)
def layer_norm(x, gamma, beta, epsilon=1e-5):
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
    x_norm = (x - mean) / tf.sqrt(variance + epsilon)
    return gamma * x_norm + beta

class CNN(tf.keras.Model):
    def __init__(self, num_classes=10, norm_type='batch'):
        super(CNN, self).__init__()
        self.norm_type = norm_type
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', use_bias=True)

        if self.norm_type == 'batch':
            self.gamma_bn = tf.Variable(tf.ones([1, 1, 1, 32]), trainable=True)
            self.beta_bn = tf.Variable(tf.zeros([1, 1, 1, 32]), trainable=True)
        elif self.norm_type == 'layer':
            self.gamma_ln = tf.Variable(tf.ones([32]), trainable=True)
            self.beta_ln = tf.Variable(tf.zeros([32]), trainable=True)

        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
        self.flatten = tf.keras.layers.Flatten()

        # Conditional initialization of the dense layer or its components for weight normalization
        if self.norm_type != 'weight':
            self.dense = tf.keras.layers.Dense(num_classes)
        else: # For 'weight' normalization
            input_to_dense_dim = 14 * 14 * 32 # (28/2) * (28/2) * 32
            self.v_dense_kernel = tf.Variable(
                tf.keras.initializers.GlorotUniform()(shape=[input_to_dense_dim, num_classes]),
                trainable=True, name='v_dense_kernel'
            )
            self.g_dense_scalar = tf.Variable(
                tf.keras.initializers.Ones()(shape=[num_classes,]),
                trainable=True, name='g_dense_scalar'
            )
            self.b_dense_bias = tf.Variable(
                tf.keras.initializers.Zeros()(shape=[num_classes,]),
                trainable=True, name='b_dense_bias'
            )

    def call(self, x, training=False):
        x = self.conv1(x)
        if self.norm_type == 'batch':
            x = batch_norm(x, self.gamma_bn, self.beta_bn)
        elif self.norm_type == 'layer':
            shape = tf.shape(x)
            x_reshaped = tf.reshape(x, [-1, x.shape[-1]])
            x = tf.reshape(layer_norm(x_reshaped, self.gamma_ln, self.beta_ln), shape)
        # If self.norm_type is 'none' or 'weight', no normalization is applied here

        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.flatten(x)

        # Conditional dense layer application
        if self.norm_type == 'weight':
            normalized_kernel = weight_norm(self.v_dense_kernel, self.g_dense_scalar, axis=0)
            x = tf.matmul(x, normalized_kernel) + self.b_dense_bias
        else:
            x = self.dense(x)

        return x

# Training setup and step function (copied from previous cell)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# The optimizer is instantiated inside train_and_evaluate_model for each model now

@tf.function
def train_step(model, images, labels):
    # This train_step function is actually not directly used by train_and_evaluate_model, but kept for context
    # The gradient tape logic is duplicated in the helper function.
    optimizer_inner = tf.keras.optimizers.Adam() # Local optimizer if this function were directly used
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer_inner.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Data Preparation (copied from previous cell to ensure datasets are available)
#batch size
batch_size = 100

# Loading and preparing the Fashion MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = np.expand_dims(x_train.astype(np.float32) / 255.0, -1)
x_test = np.expand_dims(x_test.astype(np.float32) / 255.0, -1)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

def train_and_evaluate_model(model, epochs, train_dataset, test_dataset, model_name):
    print(f"\nTraining {model_name} model...")

    # Re-initialize optimizer for each model to ensure independent training
    optimizer_model = tf.keras.optimizers.Adam()

    for epoch in range(epochs):
        # Re-initialize metrics at the beginning of each epoch to effectively reset them
        train_loss_metric_epoch = tf.keras.metrics.Mean(name='train_loss')
        test_accuracy_metric_epoch = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        for images, labels in train_dataset:
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer_model.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss_metric_epoch.update_state(loss)

        for test_images, test_labels in test_dataset:
            predictions = model(test_images, training=False)
            test_accuracy_metric_epoch.update_state(test_labels, predictions)

        print(f"Epoch {epoch + 1}, Loss: {train_loss_metric_epoch.result():.4f}, Test Accuracy: {test_accuracy_metric_epoch.result():.4f}")

    final_accuracy = test_accuracy_metric_epoch.result().numpy()
    print(f"{model_name} Final Test Accuracy: {final_accuracy:.4f}")
    return final_accuracy

# Instantiate the models
model_none = CNN(norm_type='none')
model_bn = CNN(norm_type='batch')
model_ln = CNN(norm_type='layer')
model_wn = CNN(norm_type='weight')

# Define epochs
epochs = 5

# Store results
accuracies = {}

# Train and evaluate each model
accuracies['Baseline'] = train_and_evaluate_model(model_none, epochs, train_dataset, test_dataset, 'Baseline')
accuracies['Batch Norm'] = train_and_evaluate_model(model_bn, epochs, train_dataset, test_dataset, 'Batch Norm')
accuracies['Layer Norm'] = train_and_evaluate_model(model_ln, epochs, train_dataset, test_dataset, 'Layer Norm')
accuracies['Weight Norm'] = train_and_evaluate_model(model_wn, epochs, train_dataset, test_dataset, 'Weight Norm')

print("\n--- Final Accuracies ---")
for norm_type, accuracy in accuracies.items():
    print(f"{norm_type}: {accuracy:.4f}")
