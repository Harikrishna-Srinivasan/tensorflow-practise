# MNIST Training Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harikrishna-Srinivasan/tensorflow-practise/blob/main/tf_mnist.ipynb)

## Train the Mnist dataset

### Import dependencies

```python
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
```

### Load the dataset

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```

### Set a random state for reproducibility

```python
tf.random.set_seed(101)
random.seed(101)
np.random.seed(101)
```

### Visualize a few from training data

```python
height, width = x_train[0].shape

plt.figure(figsize=(6, 10))
for i in range(20):
    plt.subplot(5, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i])
    plt.xlabel(y_train[i])
plt.show()
```

### Create a model

```python
model1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(height, width)),
    tf.keras.layers.Dense(64, activation="sigmoid"),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

### Compile the Model

```python
model1.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])
model1.summary()
```

### Fit / Train the model

```python
model1.fit(x_train, y_train, epochs=10) # accuracy=0.9036, loss=0.3087
```

### Normalizing input data

```python
x_train_normal = x_train.astype("float32") / 255.0
x_test_normal = x_test.astype("float32") / 255.0
```

```python
model2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(height, width)),
    tf.keras.layers.Dense(64, activation="sigmoid"),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(10, activation="softmax")
])
model2.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])
model2.summary()
model2.fit(x_train_normal, y_train, epochs=10) # accuracy=0.9735, loss=0.0809
```

### Test Model-1 (Unnormalized)

```python
model1.evaluate(x_test, y_test, verbose=1) # accuracy=0.92, loss=0.2355
```

### Test Model-2 (Normalized)

```python
model2.evaluate(x_test_normal, y_test, verbose=1) # accuracy=0.9766, loss=0.0789
```

## Visualize failed tests

```python
def visualize_fails(model, x_test, y_test, count=5):
    import matplotlib.pyplot as plt
    import numpy as np
    k = 0
    plt.figure(figsize=(8, 10))
    for (x, y) in zip(x_test, y_test):
        x_with_batch = np.expand_dims(x, axis=0)
        y_pred = np.argmax(model.predict(x_with_batch, verbose=0)[0])
        if y != y_pred:
            plt.subplot(5, 5, k+1)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(y)
            plt.ylabel(y_pred)
            plt.imshow(x)
            k += 1
        if k == count:
            break
    plt.show()
```

### On what tests did Model-1 fail?

```python
visualize_fails(model1, x_test, y_test)
```

### On what tests did Model-2 fail?

```python
visualize_fails(model2, x_test_normal, y_test, count=3) # quite reasonable
```

# Train model-2 for longer

```python
model2.fit(x_train_normal, y_train, verbose=1, epochs=5) # accuracy=0.9788, loss=0.064
model2.evaluate(x_test_normal, y_test) # accuracy=0.979, loss=0.7176
visualize_fails(model2, x_test_normal, y_test)
```

# Save the models

```python
model1.save("model1.h5")
model2.save("model2.h5")
```
