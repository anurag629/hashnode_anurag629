# Gender Determination with Morphometry of Eyes using deep learning model.

In this tutorial, we will be using deep learning techniques to determine the gender of a person based on the morphometry of their eyes. We will start by importing the necessary modules including TensorFlow, Keras, and Matplotlib. Next, we will define some variables such as image size, batch size, and the number of channels.

We will then use the image\_dataset\_from\_directory function to load our dataset. This function allows us to easily load images from a directory and organize them into a dataset that we can use for training and testing our model. We will also visualize some of the images in the dataset to get a better understanding of what we are working with.

After loading the dataset, we will split it into train, validation, and test sets using a custom function that we define. This is a crucial step as we will use the train set for training our model, the validation set for evaluating the model during training, and the test set for evaluating the model's performance on unseen data.

Next, we will prepare our dataset for training by applying some transformations such as shuffling, caching, and prefetching. This helps to improve the efficiency and speed of our training process.

We will then build our model using the Keras Sequential API. This allows us to easily stack layers of different types to create our model. In this case, we will use a combination of convolutional and dense layers.

After building the model, we will compile it by specifying the loss function and optimization algorithm that we want to use. We will also specify some metrics that we want to track during training.

With our model compiled, we are now ready to start training. We will do this by calling the fit function and passing it to our training and validation datasets, as well as the number of epochs that we want to train for.

Finally, we will evaluate the model's performance on the test set and visualize the results using Matplotlib. By following these steps, we should be able to create a model that can accurately determine the gender of a person based on the morphometry of their eyes.

Here is an example of how you can classify Gender with Morphometry of Eyes using a deep learning model in Python using the TensorFlow library:

# 1\. Importing modules

```python
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from tensorflow import keras
print("Done!")
```

# 2\. Defining variables

```python
image_size = 224
batch_size =32
channels = 3
epoches = 24
print("Done!")
```

# 3\. loading dataset using image\_datset\_from\_directory

`image_dataset_from_directory` is a function in TensorFlow that can be used to create a [`tf.data`](http://tf.data)`.Dataset` object for images stored in a directory structure. It can be used to load and preprocess the images for training, evaluation, or prediction.

```python
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '/kaggle/input/gender-determination-with-morphometry-of-eyes/train',
    shuffle = True,
    image_size = (image_size, image_size),
    batch_size = batch_size
)
print("Done!")
```

## 4\. Printing class names

```python
class_names = dataset.class_names
class_names
print("Done!")
```

# 5\. visualizing images

```python
plt.figure(figsize=(10, 10))
for image_batch, label_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3, 4, i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
print("Done!")
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1672113987140/31e5809b-9f5a-45bc-a115-50466e5910ff.png align="center")

# 6\. splitting our dataset into train, val, and test

```python
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size = 10000):
    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed = 12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split*ds_size)

    train_ds = ds.take(train_size)

    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

print("Done!")
```

```python
train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
print("Done!")
```

```python
print("Len train_set = ", len(train_ds))
print("Len val_set = ", len(val_ds))
print("Len test_set = ", len(test_ds))

print("Done!")
```

```python
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

print("Done!")
```

# 7\. resizing and rescaling the image

```python
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(image_size, image_size),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])

print("Done!")
```

# 8\. applying data augmentation

```python
data_augumentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
])

print("Done!")
```

# 9\. building our model

```python
# Load a pre-trained model
base_model = keras.applications.ResNet50(weights='imagenet', input_shape=(224, 224, 3))

# Remove the final layers of the model
base_model.layers.pop()
base_model.layers.pop()

# Add your own layers
model = keras.Sequential([
    resize_and_rescale, 
    data_augumentation,
    base_model,
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Freeze the weights of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("First fit started!")
# Train the model using the pre-trained model as a starting point
model.fit(train_ds,
          batch_size=32,
          epochs=5,
          validation_data=val_ds)
print("First fit done!")

# Fine-tune the model by unfreezing some layers
for layer in base_model.layers[:10]:
    layer.trainable = True

# Recompile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


print("First fit started!")
# Continue training the model
history = model.fit(train_ds,
                    batch_size=32,
                    epochs=5,
                    validation_data=val_ds,
                    verbose = 1
                   )

print("First fit done!")
```

```python
model.summary()

print("Done!")
```

# 10\. Plotting training and validation accuracy and loss

```python
# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```

# 11\. Now we will test the model for the given test data set

```python
# Set the image size and batch size
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Create an ImageDataGenerator object for the images
image_generator = ImageDataGenerator(rescale=1./255)

# Load the images from the folder
data_generator = image_generator.flow_from_directory(
    '/kaggle/input/gender-determination-with-morphometry-of-eyes/test',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,  # Do not generate labels
    shuffle=False)  # Do not shuffle the images

# Predict the class of each image
predictions = model.predict(data_generator, steps=len(data_generator))

# Get the class labels
class_labels = data_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}  # Invert the dictionary

# Print the predictions
for i, prediction in enumerate(predictions):
    print(f'Image {i+1}: {class_labels[np.argmax(prediction)]}')
```

# Thanks for liking it.......