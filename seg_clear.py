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
plt.imshow(blurred_mask, cmap='gray')
plt.title("Processed Mask")
plt.show()

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

plt.imshow(thick_skeleton, cmap='gray')
plt.title("Thick Skeleton")
plt.show()

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


plt.imshow(line_image, cmap='gray')
plt.title("Filtered Junctions")
plt.show()


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

plt.imshow(skeleton_color)
plt.title("Detected Path")
plt.show()

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
plt.figure(figsize=(8, 4))
plt.imshow(skeleton_last, cmap='gray')  # cmap='gray' provides black and white display
plt.title("Last Skeleton")
plt.axis("off")
plt.show()


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
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(skeleton_color, cv2.COLOR_BGR2RGB)) 
plt.title("Junctions Detected")
plt.axis("off")
plt.show()

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
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(final_skeleton, cv2.COLOR_BGR2RGB))  
plt.title("Last Junctions Result")
plt.axis("off")
plt.show()

'''
---------------------------------------------------------------- JUNCTION AND PATH NAMING ---------------------------------------------------------------------------------------
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
---------------------------------------------------------BURAYA KADAR GAYET IYI CALISIYOR BUNDAN SONRA YOL ISIMLENDIRME SIKINTILI------------------------------------------------------
'''
'''
8. Parsing and Naming Paths
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Minimum beyaz piksel uzunluğu (örneğin 5)
min_white_pixel_length = 25 

# Görselleştirme için orijinal görüntüye zarar vermemek adına geçici bir kopya oluştur
labeled_image = skeleton_color.copy()  # Eğer skeleton_color zaten renkli ise dönüşüm yapmaya gerek yok

# Çizgileri çizecek geçici bir görüntü
temp_image = np.zeros_like(skeleton_color)  # Geçici görüntü, yalnızca çizgileri içerecek

road_labels = []
road_counter = 0

# Junctionlar arası çizgileri çizen fonksiyon
def draw_line_between_junctions(img, start, end):
    """Çizilen iki junction noktası arasındaki çizgiyi img üzerine çizer."""
    cv2.line(img, start, end, color=(255, 255, 255), thickness=1)  # Beyaz çizgi

# Beyaz piksel kontrolü yapan fonksiyon
def is_sufficient_white_pixels(img1, img2, start, end, min_length):
    """İki görüntü arasında çizilen çizgilerde yeterli beyaz piksel olup olmadığını kontrol et."""
    # Boş bir maske oluştur
    mask = np.zeros_like(img1, dtype=np.uint8)
    
    # İki junction arasında bir çizgi çiz
    cv2.line(mask, start, end, color=255, thickness=1)
    
    # AND işlemi: İki görüntüde de beyaz piksel olup olmadığını kontrol et
    intersection = cv2.bitwise_and(img1, mask)
    intersection = cv2.bitwise_and(intersection, img2)

    # Renkli görüntüyü gri tonlamalıya dönüştür
    gray_white_pixels = cv2.cvtColor(intersection, cv2.COLOR_BGR2GRAY)
    
    # Beyaz piksel sayısını hesapla
    count = cv2.countNonZero(gray_white_pixels)
    
    # Eğer beyaz piksel sayısı eşik değeri aşarsa True döner
    return count >= min_length

# Tüm junctionlar arasındaki bağlantıları kontrol et
for i, (x1, y1) in enumerate(merged_junction_points, start=1):
    for j, (x2, y2) in enumerate(merged_junction_points[i:], start=i + 1):
        # Çizgi çiz ve beyaz piksel kontrolü yap
        draw_line_between_junctions(temp_image, (x1, y1), (x2, y2))
        
        # Eğer çizgi üzerinde yeterli beyaz piksel varsa ve skeleton_color ile kesişiyorsa
        if is_sufficient_white_pixels(skeleton_color, temp_image, (x1, y1), (x2, y2), min_white_pixel_length):
            # Yeterli beyaz piksel varsa, bu yolu geçerli kabul et ve isimlendir
            road_counter += 1
            road_label = f"R-N({i}-{j})-{road_counter}"
            road_labels.append((road_label, ((x1 + x2) // 2, (y1 + y2) // 2)))  # Ortaya etiket ekle
            
            # Görselleştirme: Geçerli yolu çiz ve etiketle
            position = ((x1 + x2) // 2 + 10, (y1 + y2) // 2 + 10)  # Ortada etiketin konumunu belirle
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            color = (255, 0, 0)  # Mavi renk yol için
            cv2.putText(labeled_image, road_label, position, font, font_scale, color, font_thickness, cv2.LINE_AA)

# Junction isimlerini görselleştir
for i, (x, y) in enumerate(merged_junction_points, start=1):
    label = f"N-{i}"
    position = (x + 15, y)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    color = (0, 255, 0)  # Yeşil renk junctionlar için
    cv2.putText(labeled_image, label, position, font, font_scale, color, font_thickness, cv2.LINE_AA)

# Sonuçları görselleştir
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB))
plt.title("Junctions and Valid Roads")
plt.axis("off")
plt.show()

# Geçerli yolların etiketlerini yazdır
print("Valid Road Labels:")
for label, (mid_x, mid_y) in road_labels:
    print(f"{label} - Location: ({mid_x}, {mid_y})")

