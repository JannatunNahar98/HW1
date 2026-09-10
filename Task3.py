import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Create a function to build the model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    return model


# -------------------------
# Model 1: Adam
# -------------------------

model_adam = create_model()

model_adam.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_adam = model_adam.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.2,
    verbose=1
)


# -------------------------
# Model 2: SGD
# -------------------------

model_sgd = create_model()

model_sgd.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_sgd = model_sgd.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.2,
    verbose=1
)


# -------------------------
# Compare Accuracy
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history_adam.history["accuracy"],
    label="Adam Training Accuracy"
)

plt.plot(
    history_adam.history["val_accuracy"],
    label="Adam Validation Accuracy"
)

plt.plot(
    history_sgd.history["accuracy"],
    label="SGD Training Accuracy"
)

plt.plot(
    history_sgd.history["val_accuracy"],
    label="SGD Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Adam vs SGD Accuracy")
plt.legend()

plt.show()