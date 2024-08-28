from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

# Step 1: Read the input image using PIL and ensure it's in RGB mode
input_image_path = 'img.png'  # Replace with your image path
image = Image.open(input_image_path)

# Convert the image to RGB if it's not already in that mode
if image.mode != 'RGB':
    image = image.convert('RGB')

image_rgb = np.array(image)

# Step 2: Reshape the image to a 2D array of pixels (RGB values)
pixels = image_rgb.reshape((-1, 3))

# Step 3: Initialize k random centroids
def initialize_centroids(pixels, k):
    random_indices = random.sample(range(len(pixels)), k)
    centroids = pixels[random_indices]
    return centroids

# Step 4: Assign each pixel to the nearest centroid
def assign_pixels_to_centroids(pixels, centroids):
    assignments = []
    for pixel in pixels:
        distances = np.linalg.norm(pixel - centroids, axis=1)
        nearest_centroid = np.argmin(distances)
        assignments.append(nearest_centroid)
    return np.array(assignments)

# Step 5: Update the centroids by calculating the mean of assigned pixels
def update_centroids(pixels, assignments, k):
    new_centroids = []
    for i in range(k):
        assigned_pixels = pixels[assignments == i]
        if len(assigned_pixels) == 0:  # Avoid division by zero
            new_centroid = pixels[random.randint(0, len(pixels)-1)]
        else:
            new_centroid = np.mean(assigned_pixels, axis=0)
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

# Step 6: Perform the k-means clustering
def kmeans(pixels, k, max_iterations=100):
    centroids = initialize_centroids(pixels, k)
    for _ in range(max_iterations):
        assignments = assign_pixels_to_centroids(pixels, centroids)
        new_centroids = update_centroids(pixels, assignments, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, assignments

# Step 7: Run the k-means algorithm
k = 5  # Number of clusters (feature colors)
centroids, assignments = kmeans(pixels, k)

# Step 8: Sort the centroids (colors) from light to dark
def luminance(color):
    # Calculate luminance using the formula: 0.299*R + 0.587*G + 0.114*B
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]

# Sort centroids by luminance
sorted_centroids = sorted(centroids, key=luminance)

# Convert sorted centroids back to a NumPy array
sorted_centroids = np.array(sorted_centroids)

# Step 9: Create a new image with the original image and sorted feature color rectangles
height, width, _ = image_rgb.shape
rect_height = 50
output_image = np.zeros((height + rect_height, width, 3), dtype=np.uint8)
output_image[:height, :, :] = image_rgb

# Draw rectangles with sorted feature colors
for i, color in enumerate(sorted_centroids.astype(int)):
    start_x = int(i * width / k)
    end_x = int((i + 1) * width / k)
    output_image[height:, start_x:end_x, :] = color

# Step 10: Save and display the output image
output_image = Image.fromarray(output_image.astype('uint8'))
output_image.save('output_image_sorted.jpg')

# Display the output image using matplotlib
plt.figure(figsize=(8, 8))
plt.imshow(output_image)
plt.axis('off')
plt.show()

# Print the RGB values of the sorted feature colors
print("Sorted feature colors (RGB) from light to dark:")
for i, color in enumerate(sorted_centroids.astype(int)):
    print(f"Color {i + 1}: {color}")