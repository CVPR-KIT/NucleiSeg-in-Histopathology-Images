import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read or generate the image
original_image = cv2.imread('Dataset/testNormal/0.png')  # Grayscale for simplicity
print(original_image.shape)

# Apply Max Pooling
pool_size = 3
max_pooled_image = cv2.resize(original_image, (original_image.shape[1] // pool_size, original_image.shape[0] // pool_size), interpolation=cv2.INTER_NEAREST)

# Apply Gaussian Blur
blurred_image = cv2.GaussianBlur(max_pooled_image, (5, 5), 0)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original_image, cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(max_pooled_image, cmap='gray')
axes[1].set_title('Max Pooled Image')
axes[2].imshow(blurred_image, cmap='gray')
axes[2].set_title('MaxBlurPooled Image')

for ax in axes:
    ax.axis('off')

plt.savefig("test.png")
