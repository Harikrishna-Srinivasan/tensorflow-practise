import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

tf.random.set_seed(101)
random.seed(101)
np.random.seed(101)

# Dimensions of X
height, width = x_train[0].shape

plt.figure(figsize=(6, 10))
for i in range(20):
    plt.subplot(5, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i])
    plt.xlabel(y_train[i])
plt.show()

model1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(height, width)),
    tf.keras.layers.Dense(64, activation="sigmoid"),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])

model1.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

model1.summary()

model1.fit(x_train, y_train, epochs=10) # accuracy=0.9036, loss=0.3087

x_train_normal = x_train.astype("float32") / 255.0
x_test_normal = x_test.astype("float32") / 255.0

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

model1.evaluate(x_test, y_test, verbose=1) # accuracy=0.92, loss=0.2355

model2.evaluate(x_test_normal, y_test, verbose=1) # accuracy=0.9766, loss=0.0789

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

visualize_fails(model1, x_test, y_test)

visualize_fails(model2, x_test_normal, y_test, count=3) # quite reasonable

model2.fit(x_train_normal, y_train, verbose=1, epochs=5) # accuracy=0.9788, loss=0.064

model2.evaluate(x_test_normal, y_test) # accuracy=0.979, loss=0.7176

visualize_fails(model2, x_test_normal, y_test)

model1.save("model1.h5")
model2.save("model2.h5")
