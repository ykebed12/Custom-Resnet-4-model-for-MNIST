import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import datasets
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import activations
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os

# Set a global random seed for TensorFlow
tf.random.set_seed(42)
np.set_printoptions(suppress=True)   # suppress scientific notation

# Create Folder if it doesnt exist
if not os.path.exists('output'):
  os.makedirs('output')


for folder in ["output/train", "output/valid", "output/test"]:
  if not os.path.exists(folder):
    os.makedirs(folder)


## Prepare data

mnist = tf.keras.datasets.mnist
(x_train, label_train), (x_test, label_test) = mnist.load_data()

num_train = x_train.shape[0]
num_test  = x_test.shape[0]

# One-hot encode the training
y_train = np.zeros([num_train, 10])
for i in range(num_train):
  y_train[i, label_train[i]] = 1

# One-hot encode the testing
y_test  = np.zeros([num_test, 10])
for i in range(num_test):
	y_test[i, label_test[i]] = 1

## select train, validation and test data as per the project description
train_size = 4000
validation_size = 1000
test_size = 1000

my_x_train = x_train[:train_size]
my_y_train = y_train[:train_size]
my_x_valid = x_train[train_size:train_size+validation_size]
my_y_valid = y_train[train_size:train_size+validation_size]
my_x_test = x_test[:test_size]
my_y_test = y_test[:test_size]

for folder, dataset in (("train", my_x_train), ("valid", my_x_valid), ("test", my_x_test)):
  for i in range(5):
    savePath = "output/"+folder+"/"+str(i)+".png"
    cv2.imwrite(savePath, dataset[i])

my_x_train, my_x_valid, my_x_test = my_x_train / 255.0, my_x_valid / 255.0, my_x_test / 255.0


#####################################
## DATA PREPARATION
batch_size = 50
# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((my_x_train, my_y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((my_x_valid, my_y_valid))
val_dataset = val_dataset.batch(batch_size)


############################
# Create the model
#--------------------------

def myModel():

    # FIRST SECTION
    inputs = keras.Input(shape=(28, 28, 1), name="Input_1")
    data_augmentation = tf.keras.Sequential([
        RandomRotation(0.1)  # Randomly rotate images by up to 20 degrees
    ])
    x = data_augmentation(inputs)
    x = layers.Conv2D(5, (3, 3), padding="same", activation='relu', name="conv_11", kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(inputs)
    x = layers.Conv2D(5, (3, 3), padding="same", activation='relu', name="conv_12", kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
    add_1 = layers.Add()([x, inputs])

    # # # Second Section
    maxPool1 = layers.MaxPooling2D(pool_size=(2,2), strides=2)(add_1)
    x = layers.Conv2D(5, (3,3), padding="same", activation='relu', kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(maxPool1)
    add_2 = layers.Add()([x, maxPool1])

    flat = layers.Flatten()(add_2)
    dense = layers.Dense(10, activation="softmax")(flat)
    model = keras.Model(inputs=inputs, outputs=dense, name="My_Model")
    return model

model = myModel()
print(model.summary())

######################################
## Train

# Prepare the metrics.
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()
test_acc_metric = keras.metrics.CategoricalAccuracy()

# Instantiate an optimizer.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)


def training_loop (train_dataset, val_dataset, model, num_epochs = 150):
  train_acc_results = []
  val_acc_results = []
  train_loss_results = []
  val_loss_results = []

  for epoch in range(num_epochs):
    print("\nStart of epoch %d" % (epoch,))

    epoch_loss = []
    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

      # Open a GradientTape to record the operations run
      # during the forward pass, which enables auto-differentiation.
      with tf.GradientTape() as tape:
          # Run the forward pass of the layer.
          # The operations that the layer applies
          # to its inputs are going to be recorded
          # on the GradientTape.
          logits = model(x_batch_train, training=True)  # Logits for this minibatch
          # Compute the loss value for this minibatch.
          loss_value = loss_fn(y_batch_train, logits)

      train_acc_metric.update_state(y_batch_train, logits)
      # Use the gradient tape to automatically retrieve
      # the gradients of the trainable variables with respect to the loss.
      grads = tape.gradient(loss_value, model.trainable_weights)
      # Run one step of gradient descent by updating
      # the value of the variables to minimize the loss.
      optimizer.apply_gradients(zip(grads, model.trainable_weights))
      epoch_loss.append(loss_value)
    
    train_loss_results.append(sum(epoch_loss) / len(epoch_loss))
    

    train_acc = train_acc_metric.result()
    train_acc_metric.reset_states()
    print("Train acc: %.4f" % (float(train_acc),))

    train_acc_results.append(float(train_acc))

    val_batch_loss = []
    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val, training=False)
        # Update val metrics
        val_acc_metric.update_state(y_batch_val, val_logits)
        val_batch_loss.append(loss_fn(y_batch_val, val_logits))
    avg_val_loss = sum(val_batch_loss) / len(val_batch_loss)
    val_loss_results.append(avg_val_loss)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Train loss: %.4f" % (float(sum(epoch_loss) / len(epoch_loss)),))
    print("Validation loss: %.4f" % (float(avg_val_loss),))

    val_acc_results.append(float(val_acc))
  
    
  return (train_acc_results, val_acc_results, train_loss_results, val_loss_results)

dot_img_file = './model_graph.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
train_acc_results, val_acc_results, train_loss_results, val_loss_results = training_loop (train_dataset, val_dataset, model, num_epochs=150)


##########################
# 
plt.plot(train_loss_results, label = "train")
plt.plot(val_loss_results, label = "valid")
plt.legend()

plt.xlabel('epochs', fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.title('Resnet-4 Loss')
# Save plot
plt.grid()
plt.savefig('output/loss.png')

plt.show()

plt.plot(train_acc_results, label = "train")
plt.plot(val_acc_results, label = "valid")
y_predict = model(x_test, training=False)
test_acc_metric.update_state(y_test, y_predict)
y_predict_acc = test_acc_metric.result()
plt.plot([y_predict_acc]*len(val_acc_results), label="test")
plt.legend()

plt.xlabel('epoch', fontsize=14)
plt.ylabel('accuracy', fontsize=14)
plt.title('Resnet-4 Loss')
# Save plot
plt.grid()
plt.savefig('output/accuracy.png')

plt.show()

