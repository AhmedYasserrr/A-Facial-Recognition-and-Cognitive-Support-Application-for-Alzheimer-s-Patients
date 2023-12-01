import matplotlib.pyplot as plt

# Past results (replace with your actual data)
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
recall_values = [0.6058, 0.6703, 0.8014, 0.8223, 0.855, 0.85, 0.8957, 0.8723, 0.9097, 0.9309, 0.9531, 0.9496, 0.9375, 0.9571, 0.9669]

# Plotting the confidence curve
plt.figure(figsize=(8, 6))
plt.plot(epochs, recall_values, marker='o', linestyle='-')

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.title('Recall Confidence Curve')

# Adding a grid for better readability
plt.grid(True)

# Display the plot
plt.show()
import matplotlib.pyplot as plt

# Past results for Precision (replace with your actual data)
precision_values = [0.6587, 0.7922, 0.904, 0.8939, 0.9465, 0.9482, 0.9651, 0.9755, 0.9655, 0.966, 0.967, 0.9888, 0.9846, 0.9926, 0.985]

# Plotting the confidence curve for Precision
plt.figure(figsize=(8, 6))
plt.plot(epochs, precision_values, marker='o', linestyle='-')

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.title('Precision Confidence Curve')

# Adding a grid for better readability
plt.grid(True)

# Display the plot
plt.show()
import matplotlib.pyplot as plt

# Past results for Loss (replace with your actual data)
loss_values = [0.55682015, 0.3717262, 0.45546362, 0.19804838, 0.56170523, 0.36590827, 0.26314503, 0.043505292, 0.26257378, 0.22667147, 0.02511103, 0.18793643, 0.10799833, 0.103368476, 0.036094315]

# Plotting the confidence curve for Loss
plt.figure(figsize=(8, 6))
plt.plot(epochs, loss_values, marker='o', linestyle='-')

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Confidence Curve')

# Adding a grid for better readability
plt.grid(True)

# Display the plot
plt.show()
