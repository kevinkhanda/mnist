import keras
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[i]))

model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=keras.activations.relu),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation=keras.activations.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)
model.evaluate(X_test, y_test)
