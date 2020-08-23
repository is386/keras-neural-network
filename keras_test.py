import time
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

NAME = "cnn-{}".format(int(time.time()))

TC = 15
LR = 0.001
BATCH_SIZE = 8


def train(x, y):
    # Shows loss and accuracy graphs in browser
    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

    # Base Model
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(128, (5, 5), input_shape=x.shape[1:]))

    # Max Pooling Layer 1
    model.add(MaxPooling2D(2, 2))

    # Convolutional Layer 2
    model.add(Conv2D(64, (5, 5)))

    # Max Pooling Layer 2
    model.add(MaxPooling2D(2, 2))

    # Flattening Layer
    model.add(Flatten())

    # Hidden Layer
    model.add(Dense(32, activation="relu"))

    # Output Layer
    model.add(Dense(16, activation="softmax"))

    # Compiles the model with a loss function and optimizer
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=LR),
        metrics=["accuracy"],
    )

    # Displays a summary of the model
    model.summary()

    # Training
    model.fit(
        x, y,
        epochs=TC,
        batch_size=BATCH_SIZE,
        callbacks=[tensorboard],
        validation_split=0.33
    )
    return model


def test(x, y, model):
    return model.evaluate(x, y, batch_size=BATCH_SIZE)[1] * 100


def main():
    x_train = pickle.load(open("train/x.pickle", "rb"))
    y_train = pickle.load(open("train/y.pickle", "rb"))
    x_test = pickle.load(open("test/x.pickle", "rb"))
    y_test = pickle.load(open("test/y.pickle", "rb"))

    x_train = keras.utils.normalize(x_train, axis=1)
    x_test = keras.utils.normalize(x_test, axis=1)

    model = train(x_train, y_train)
    acc = test(x_test, y_test, model)

    print("Accuracy:", acc)


if __name__ == "__main__":
    main()
