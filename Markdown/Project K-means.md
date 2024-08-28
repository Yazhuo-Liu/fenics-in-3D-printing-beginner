
# K-Means Clustering: A Beginner's Tutorial

## Introduction

K-means clustering is one of the most popular unsupervised machine learning algorithms. It is used to partition a dataset into `k` distinct clusters, where each data point belongs to the cluster with the nearest mean (centroid). This tutorial will walk you through the mathematical foundation, general workflow, and a Python implementation of the k-means algorithm without using advanced libraries.

#### Convergence of naive k-means
![Convergence of k-means](/imgs/K-means_convergence.gif)

## Mathematical Foundation

### The Objective

The goal of k-means is to minimize the within-cluster variance, which is the sum of squared distances between each data point and the centroid of its assigned cluster. The objective function is defined as:

$$
J = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2
$$

where:
- $k$ is the number of clusters.
- $C_i$ is the set of points in cluster $i$.
- $\mu_i$ is the centroid (mean) of cluster $i$.
- $\|x - \mu_i\|^2$ is the squared Euclidean distance between point $x$ and centroid $\mu_i$.

### The Algorithm

1. **Initialization**: Select $k$ initial centroids randomly from the dataset.
2. **Assignment**: Assign each data point to the nearest centroid.
3. **Update**: Recalculate the centroids as the mean of all data points assigned to each cluster.
4. **Convergence**: Repeat the assignment and update steps until the centroids no longer change significantly.

## General Workflow

1. **Choose the Number of Clusters ($k$)**:
   - Determine the number of clusters you want the algorithm to find in the dataset.

2. **Initialize the Centroids**:
   - Randomly select $k$ data points as the initial centroids.

3. **Assign Data Points to Nearest Centroid**:
   - Calculate the distance between each data point and each centroid.
   - Assign each data point to the cluster with the nearest centroid.

4. **Update Centroids**:
   - Recalculate the centroids by computing the mean of all data points in each cluster.

5. **Repeat Until Convergence**:
   - Repeat the assignment and update steps until the centroids stabilize.

## Project description
In this project, you will apply the k-means algorithm to an image to identify the most prominent colors (clusters) within it. The RGB values of each pixel will be treated as data points in a three-dimensional space. After clustering the pixels using K-means, the resulting colors will be sorted from light to dark based on their luminance.

### How to Compute Lightness and Darkness of a Color

When working with colors, especially in tasks like image processing or design, it's often useful to determine whether a color is "light" or "dark." This can be computed using the RGB values of the color.

### Understanding RGB Values

Every color on a digital screen is composed of three components: Red (R), Green (G), and Blue (B). Each of these components typically has a value between 0 and 255.

- **Red (R)**: Intensity of the red component
- **Green (G)**: Intensity of the green component
- **Blue (B)**: Intensity of the blue component

The combination of these three components determines the final color.

### Calculating Luminance

One common way to compute the "lightness" or "darkness" of a color is by calculating its luminance. Luminance is a weighted sum of the RGB components, reflecting the human eye's sensitivity to these colors.

### Luminance Formula

The standard formula for calculating luminance is:

$$
\text{Luminance} = 0.299 * R + 0.587 * G + 0.114 * B
$$


### Why These Weights?

- **Red (R)**: The human eye is more sensitive to green light than red or blue. That's why green has the highest weight in this formula.
- **Green (G)**: Green contributes the most to luminance due to its high sensitivity to the human eye.
- **Blue (B)**: Blue contributes the least to luminance, as the human eye is less sensitive to blue light.

### Interpretation of Luminance

- **Higher Luminance Values**: Indicate that the color is lighter.
- **Lower Luminance Values**: Indicate that the color is darker.

### Example Calculation

Let’s say you have a color with the following RGB values:

- R = 50
- G = 100
- B = 150

The luminance of this color would be calculated as:

$$
\text{Luminance} = 0.299 * 50 + 0.587 * 100 + 0.114 * 150 = 14.95 + 58.7 + 17.1 = 90.75
$$

With a luminance of 90.75, this color would be considered relatively dark, especially compared to the maximum possible luminance of 255.

### Project Objective
Your task is to implement the k-means clustering algorithm from scratch, without using advanced machine learning libraries. The algorithm should:

- Read an image and extract the RGB values of each pixel.
- Perform k-means clustering on the RGB values to identify the dominant colors in the image.
- Sort the resulting colors from light to dark using their luminance values.
- Generate and display an output image that includes the original image and rectangles representing the sorted colors.

### Expected Outcome
By completing this task, you will gain a deeper understanding of how clustering algorithms work and how they can be applied to real-world data, such as images. You will also learn how to manipulate and analyze color data in Python, providing a foundation for more advanced image processing techniques.

**Here is the input image**:
![input](/imgs/img.png)

**Here is the output image**:
![output](/imgs/output_image_sorted.jpg)
output text:
```bash
Sorted feature colors (RGB) from light to dark:
Color 1: [24 27 58]
Color 2: [ 59  70 128]
Color 3: [128 155 195]
Color 4: [214 229 237]
Color 5: [250 250 249]
```

## Python Implementation

Here is a simple implementation of k-means clustering in Python, without using advanced libraries like scikit-learn or OpenCV.

make sure you have install `pillow` package
```bash
conda install -c conda-forge pillow
```

Please use this code as a reference, **program independently first**, and then compare it to this example

```python
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
```



## Conclusion

In this tutorial, we've covered the mathematical foundation of k-means clustering, walked through its general workflow, and provided a Python implementation for beginners. By understanding and implementing this algorithm from scratch, you can gain deeper insights into how clustering works and apply it to various data analysis tasks.

Happy coding!
