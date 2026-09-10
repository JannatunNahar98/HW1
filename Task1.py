import tensorflow as tf

# Create a random tensor of shape (4, 6)
tensor = tf.random.normal((4, 6))

print("Original Tensor:")
print(tensor)

# Find rank and shape
print("\nOriginal Rank:", tf.rank(tensor).numpy())
print("Original Shape:", tensor.shape)

# Reshape tensor to (2, 3, 4)
reshaped_tensor = tf.reshape(tensor, (2, 3, 4))

print("\nReshaped Tensor:")
print(reshaped_tensor)

print("Reshaped Rank:", tf.rank(reshaped_tensor).numpy())
print("Reshaped Shape:", reshaped_tensor.shape)

# Transpose from (2, 3, 4) to (3, 2, 4)
transposed_tensor = tf.transpose(reshaped_tensor, perm=[1, 0, 2])

print("\nTransposed Tensor:")
print(transposed_tensor)

print("Transposed Rank:", tf.rank(transposed_tensor).numpy())
print("Transposed Shape:", transposed_tensor.shape)

# Smaller tensor of shape (1, 4)
small_tensor = tf.random.normal((1, 4))

# Broadcast and add
result = transposed_tensor + small_tensor

print("\nSmall Tensor Shape:", small_tensor.shape)
print("Result Shape:", result.shape)
print("Result:")
print(result)