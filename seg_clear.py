import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
------------------------------------------------------- SEGMENTATION FILTERING ---------------------------------------------------------------------
'''

'''
1. Processing the Segmentation Result
'''

# Loading the segmentation mask
mask = cv2.imread("Photos/0.png", cv2.IMREAD_GRAYSCALE)

# Thresholding (Converting to binary image)
_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

# Gaussian Blurring
blurred_mask = cv2.GaussianBlur(cleaned_mask, (5, 5), 0)
_, blurred_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)


# Visualize the result
# plt.imshow(blurred_mask, cmap='gray')
# plt.title("Processed Mask")
# plt.show()

'''
2. Skeletonization and Edit Segmentation with Filtering and Cleaning
'''

from skimage.morphology import skeletonize

# 2.1 Skeletonization process
binary_mask = blurred_mask // 255  # Binary image is converted to 1 and 0 format
skeleton = skeletonize(binary_mask).astype(np.uint8) * 255

# 2.2 Thickening the skeleton
kernel = np.ones((5, 5), dtype=np.uint8)
thick_skeleton = cv2.dilate(skeleton, kernel, iterations=1)

# plt.imshow(thick_skeleton, cmap='gray')
# plt.title("Thick Skeleton")
# plt.show()

# 2.3 Cleaning Up False Branches
def prune_skeleton(thick_skeleton):
    h, w = thick_skeleton.shape
    pruned = thick_skeleton.copy()
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if thick_skeleton[y, x] == 255:
                neighbors = thick_skeleton[y - 1:y + 2, x - 1:x + 2].sum() // 255 - 1
                # Clean up unnecessary branches
                if neighbors > 3:
                    pruned[y, x] = 0
    return pruned

pruned_skeleton = prune_skeleton(thick_skeleton)

# 2.4 Filtering Linear Paths
lines = cv2.HoughLinesP(thick_skeleton, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)      # Hough Transform

# Joining lines
line_image = np.zeros_like(thick_skeleton)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), 255, 1)


# plt.imshow(line_image, cmap='gray')
# plt.title("Filtered Junctions")
# plt.show()


'''
3. Filling in Missing Pixels on Linear Paths
'''

def find_path(line_image):
    path_part = []
    h, w = line_image.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if line_image[y, x] == 255:
                # Check all 8 pixels around a pixel
                neighbors = line_image[y-1:y+2, x-1:x+2].sum() // 255 - 1
                if neighbors >= 2:  # If there are 2 or more common pixels this is a part of the path
                    path_part.append((x, y))
    return path_part

path_points = find_path(line_image)

# Visualize the result
skeleton_color = cv2.cvtColor(line_image, cv2.COLOR_GRAY2BGR)
for x, y in path_points:
    cv2.circle(skeleton_color, (x, y), 5, (255, 255, 255), -1)

# plt.imshow(skeleton_color)
# plt.title("Detected Path")
# plt.show()

'''
----------------------------------------------------------- JUNCTION DETECTION ---------------------------------------------------------------------
'''

'''
4. Skeletonization for junction detection
'''
# 3 channel to 1 channel
if len(skeleton_color.shape) == 3: 
    skeleton_gray = cv2.cvtColor(skeleton_color, cv2.COLOR_BGR2GRAY)
else:
    skeleton_gray = skeleton_color  

# Binarize the image (black and white)
_, binary_image = cv2.threshold(skeleton_gray, 127, 255, cv2.THRESH_BINARY)

# Skeletonizing a binary image
skeleton_binary = binary_image // 255  # Convert to 0 and 1 format
skeleton_last = skeletonize(skeleton_binary).astype(np.uint8) * 255  # Back to 0-255 format

# Visualize the result
# plt.figure(figsize=(8, 4))
# plt.imshow(skeleton_last, cmap='gray')  # cmap='gray' provides black and white display
# plt.title("Last Skeleton")
# plt.axis("off")
# plt.show()


'''
5. Junctions Detection
'''

# 3 channel to 1 channel
if len(skeleton_last.shape) == 3:  
    skeleton_last_gray = cv2.cvtColor(skeleton_last, cv2.COLOR_BGR2GRAY)
else:
    skeleton_last_gray = skeleton_last  

# Function to detect junctions
def detect_junctions(skeleton_last_gray):
    height, width = skeleton_last_gray.shape
    junctions = []

    for y in range(1, height-1):
        for x in range(1, width-1):
            if skeleton_last_gray[y, x] == 255:  
                
                neighbors = skeleton_last_gray[y-1:y+2, x-1:x+2]
                # Count the number of white neighbors
                white_neighbors = np.sum(neighbors == 255) - 1  # Take yourself out

                # If at least 3 neighbors are white, it is an junction
                if white_neighbors >= 3:
                    junctions.append((x, y))

    return junctions

# Detect junctions
junction_points = detect_junctions(skeleton_last_gray)

# Convert image to RGB (for markup)
skeleton_color = cv2.cvtColor(skeleton_last_gray, cv2.COLOR_GRAY2BGR)

# Mark junctions as red dots
for x, y in junction_points:
    cv2.circle(skeleton_color, (x, y), 1, (0, 0, 255), -1)  

# Visualize the result
# plt.figure(figsize=(10, 5))
# plt.imshow(cv2.cvtColor(skeleton_color, cv2.COLOR_BGR2RGB)) 
# plt.title("Junctions Detected")
# plt.axis("off")
# plt.show()

'''
6. Correcting Faulty Junction Detections
'''

from scipy.spatial import distance

# Function to connect intersections in same areas
def merge_close_junctions(junctions, threshold=10):
    merged_junctions = []
    used = set()

    for i, point1 in enumerate(junctions):
        if i in used:
            continue
        group = [point1]
        for j, point2 in enumerate(junctions):
            if j != i and j not in used:
                if distance.euclidean(point1, point2) <= threshold:  
                    group.append(point2)
                    used.add(j)
        # Calculate group center
        group_center = np.mean(group, axis=0).astype(int)
        merged_junctions.append(tuple(group_center))
        used.add(i)

    return merged_junctions

# Connect junctions
merged_junction_points = merge_close_junctions(junction_points, threshold=15)

# Convert image to RGB (for markup)
final_skeleton = cv2.cvtColor(skeleton_last_gray, cv2.COLOR_GRAY2BGR)

# Mark merged intersections
for x, y in merged_junction_points:
    cv2.circle(final_skeleton, (x, y), 10, (0, 255, 0), -1)  

# Visualize the result
# plt.figure(figsize=(10, 5))
# plt.imshow(cv2.cvtColor(final_skeleton, cv2.COLOR_BGR2RGB))  
# plt.title("Last Junctions Result")
# plt.axis("off")
# plt.show()

'''
---------------------------------------------------------------- PARSING AND NAMING JUNCTIONS ---------------------------------------------------------------------------------------
'''

'''
7. Parsing and Naming Junctions
'''

# We assume that the variable c is in the format [(x1, y1), (x2, y2), ...]

# Naming junctions and printing output
print("Junctions:")
for i, (x, y) in enumerate(merged_junction_points, start=1):
    print(f"N-{i} - Location: ({x}, {y})")

import cv2

# Make a copy of the image
labeled_image = final_skeleton.copy()  # We are working on final_skeleton

# Printing intersection names and locations
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.4  
font_thickness = 1
color = (0, 255, 0)  


for i, (x, y) in enumerate(merged_junction_points, start=1):
    label = f"N-{i}"  # Junction Name
    position = (x + 15, y)  
    cv2.putText(labeled_image, label, position, font, font_scale, color, font_thickness, cv2.LINE_AA)
    

# Visualize the result
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)) 
plt.title("Junctions Name")
plt.axis("off")
plt.show()

'''
---------------------------------------------------------------- PARSING AND NAMING PATHS ----------------------------------------------------------
'''

'''
8. Parsing and Naming Paths
'''

# 8.1 Creating Circles at Junctions for Masking

image_path = "Photos/0.png"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Resim bulunamadı: {image_path}")

# Resmin boyutlarını al
height, width, _ = image.shape

# Siyah yüzey oluştur
black_surface = np.zeros((height, width, 3), dtype=np.uint8)

# Yeşil daireleri çiz
radius = 11
color = (255, 255, 255)  # Yeşil (BGR formatında)
thickness = 1  # Daireyi doldur

for i, (x, y) in enumerate(merged_junction_points, start=1):
    print(f"{i}. Nokta: Daire x = {x}, y = {y}")
    cv2.circle(black_surface, (x, y), radius, color, thickness)

# Visualize the result
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(black_surface, cv2.COLOR_BGR2RGB)) 
plt.title("Green Circles on Black Surface")
plt.axis("off")
plt.show()

# 8.2 Finding Roads Starting Point with Masking

if black_surface is None or labeled_image is None:
    raise FileNotFoundError("Görsellerden biri bulunamadı.")

# Görsellerin aynı boyutta olduğundan emin olun
if black_surface.shape != labeled_image.shape:
    raise ValueError("Görseller aynı boyutta olmalı!")

# Ortak pikselleri bul (Her iki görüntüde de aynı renk değerine sahip olan pikseller)
common_pixels = cv2.bitwise_and(black_surface, labeled_image)

# Visualize the result
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(common_pixels, cv2.COLOR_BGR2RGB)) 
plt.title("Common Pixels")
plt.axis("off")
plt.show()


