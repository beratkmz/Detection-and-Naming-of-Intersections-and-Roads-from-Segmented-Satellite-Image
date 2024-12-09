# Detection and Naming of Intersections and Roads from Segmented Satellite Images

<p align="center">
  <img src="https://github.com/user-attachments/assets/e2862e78-eae8-49f5-aa02-f2f2173b4b2b" width="400">
  <img src="https://github.com/user-attachments/assets/f49d7df4-da3c-4912-9692-b8d63d9f8bba" width="400">

</p>

This study is a project that will be used for our VTOL vehicle, which detects collapsed buildings in the region after an earthquake and creates a road map for the location of this building. My work in this part of our project includes the detection and naming of roads and intersections in the visual so that the segmented and road masked images can be used in creating a target road plan.

## Requirements
- An image segmentation model developed for road detection
- Black and white mask datasets obtained with this model
- Any code editor and development tools for python code
- Python libraries: `opencv-python`, `numpy`, `matplotlib`, `skimage.morphology`, and `scipy.spatial`

## Features
- Ability to work properly even with segmentation results that have a margin of error with filtering and skeletonization features
- Intersection and road location datasets for road planning applications
- Python-based implementation for ease of use and customization
- Dockerized environment for easy deployment across different systems

## Usage

This code has several steps before the final output and can be used for different purposes.

### Step 1: Segmentation Filtering
#### 1.1 Processing the Segmentation Result

In this first step, thresholding, morphological and blurring processes are performed to make the ready segmented datasets better quality.

<img src="https://github.com/user-attachments/assets/4262ad09-0880-43c3-9f0e-c2bc5a57945b" width="500">

#### 1.2 Skeletonization and Edit Segmentation with Filtering and Cleaning

This step is the process of reducing the road map to pixel size by performing a skeletonization operation on the processed mask image created in the first step.

If our image segmentation process is optimized and we get very good results, we can move on to the junction detection step directly after this step. However, if there are many errors in our model, we need to go through a few more processes before moving on to the junction detection step. 

We filter our results as much as possible by thickening the skeleton model we created and performing pixel-sized operations and deleting faulty paths. After this, we can detect the paths more clearly by using the Hough transform method to obtain linear paths. However, since errors in pixel size may occur after this process, we also add a code to fill in the intermediate pixels. We do all these operations to tolerate faulty segmentation. For this, we can proceed to the intersection detection part by performing skeletonization again.

<p align="left">
<img src="https://github.com/user-attachments/assets/a510f1e1-f79e-44cb-ab22-e49a0a0892cc" width="300">
<img src="https://github.com/user-attachments/assets/e6a2a7ce-d864-4716-9c80-b086908f0e28" width="300">
<img src="https://github.com/user-attachments/assets/d6cb6b9c-8e33-4945-976e-dd31a4dbae1e" width="300">
</p>

### Step 2: Junction Detection

The pixel logic of the juntion detection was completely thought of with the junction logic and transferred to the code. An intersection consists of the junction of at least 3 roads. If only 2 roads join an intersection, we can see it as a straight road. When we interpret this fact in pixel size, we reduce our road to a single pixel size with the skeletonization process and examine the pixels around each white pixel and if we detect at least 3 white pixels, we can define this point as an intersection.
<p align="left">
<img src="https://github.com/user-attachments/assets/fbceea79-c22d-4043-8da5-d5646c34b6dc" width="500">
</p>

Although this process provides accurate results, some pixel errors cause a margin of error. In order to reduce this margin as much as possible, we obtain a more accurate result by combining common intersection points in a certain area.
<p align="left">
<img src="https://github.com/user-attachments/assets/44a91e80-63d0-4e52-aa32-6cc940fe46b8" width="500">
</p>

### Step 3: Parsing and Naming Junctions

At this stage, the coordinates of the points that I have combined as a result of the last process are taken and they are printed on the image by naming them in a certain axis size and in a certain order.
As a result, we get an output like this:
```sh
Junctions:
N-1 - Location: (885, 60)
N-2 - Location: (621, 61)
N-3 - Location: (283, 96)
N-4 - Location: (1180, 98)
N-5 - Location: (676, 115)
N-6 - Location: (217, 129)
N-7 - Location: (1234, 146)
N-8 - Location: (220, 148)
...
```
<img src="https://github.com/user-attachments/assets/ec896652-2d98-4be0-b0cf-a258f0c33489" width="500">


## License
MIT
