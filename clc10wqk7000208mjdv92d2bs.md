# Completed

# How to save the trained model in TensorFlow?

To save a trained model in TensorFlow, follow these steps:

1. Create a `tf.keras.Model` object or subclass of it.
    
2. Train the model using the `fit()` method.
    
3. Create a `tf.keras.ModelCheckpoint` callback object and pass it to the `fit()` method as an argument.
    
4. Set the `save_weights_only` parameter to `True` in the `ModelCheckpoint` callback object. This will save only the weights of the model, not the entire model structure.
    
5. Set the `filepath` parameter in the `ModelCheckpoint` callback object to the desired file location where you want to save the model.
    
6. Run the `fit()` method to train the model and save it at the specified file location.
    

Example:

```python
Copy code# Create a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create a ModelCheckpoint callback to save the model weights
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='/tmp/model.h5', save_weights_only=True)

# Train the model and save the weights
model.fit(x_train, y_train, epochs=10, callbacks=[checkpoint])
```

You can then load the saved model weights using the `load_weights()` method of the model object:

```python
Copy code# Load the saved model weights
model.load_weights('/tmp/model.h5')

# Evaluate the model on the test data
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')
```

# Hope you like it!

# Keep coding! :)
