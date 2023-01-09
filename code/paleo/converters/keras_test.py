import tensorflow as tf
from converters.keras import convert_model

l = tf.keras.layers
model = tf.keras.Sequential([
    l.Conv2D(32, 5, padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1)),
    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
    l.Conv2D(64, 5, padding='same', activation=tf.nn.relu),
    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
    l.Flatten(),
    l.Dense(1024, activation=tf.nn.relu),
    l.Dropout(0.4),
    l.Dense(10, activation=tf.nn.softmax),
])

json = convert_model(model, 128, 'Keras MNIST CNN')

with open('nets/mnist.json', 'w') as file:
    file.write(json)

pass
