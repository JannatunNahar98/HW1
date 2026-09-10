import tensorflow as tf
import matplotlib.pyplot as plt

# True values
y_true = tf.constant([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], dtype=tf.float32)

# Initial predictions
y_pred = tf.constant([
    [0.7, 0.2, 0.1],
    [0.2, 0.6, 0.2],
    [0.1, 0.2, 0.7]
], dtype=tf.float32)

# Calculate MSE
mse = tf.reduce_mean(tf.square(y_true - y_pred))

# Calculate Categorical Cross-Entropy
cce = tf.reduce_mean(
    tf.keras.losses.categorical_crossentropy(y_true, y_pred)
)

print("Original Predictions")
print("MSE:", mse.numpy())
print("Categorical Cross-Entropy:", cce.numpy())

# Modified predictions
y_pred_modified = tf.constant([
    [0.9, 0.05, 0.05],
    [0.05, 0.9, 0.05],
    [0.05, 0.05, 0.9]
], dtype=tf.float32)

# Calculate modified losses
mse_modified = tf.reduce_mean(
    tf.square(y_true - y_pred_modified)
)

cce_modified = tf.reduce_mean(
    tf.keras.losses.categorical_crossentropy(
        y_true, y_pred_modified
    )
)

print("\nModified Predictions")
print("MSE:", mse_modified.numpy())
print("Categorical Cross-Entropy:", cce_modified.numpy())

# Plot comparison
loss_names = ["MSE", "Cross-Entropy"]

original_losses = [
    mse.numpy(),
    cce.numpy()
]

modified_losses = [
    mse_modified.numpy(),
    cce_modified.numpy()
]

x = range(len(loss_names))

plt.figure(figsize=(8, 5))

plt.bar(
    [i - 0.2 for i in x],
    original_losses,
    width=0.4,
    label="Original Predictions"
)

plt.bar(
    [i + 0.2 for i in x],
    modified_losses,
    width=0.4,
    label="Modified Predictions"
)

plt.xticks(list(x), loss_names)
plt.ylabel("Loss")
plt.title("MSE vs Categorical Cross-Entropy")
plt.legend()

plt.show()