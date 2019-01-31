# Writeup
---

**Advanced Lane Finding Project**

The goals of this project is to build a more robust lane detection algorithm that shows the forward lane location as well as providing statistics on lane curvature and vehicle offset within the lane.  The general steps are:

    * Perform once: Find camera calibration and distortion coefficients
    * Undistort the raw camera image
    * Perform color and gradient magnitude thresholding to create a binary image
    * Use a Region of Interest (ROI) filter to remove extraneous parts of the image
    * Apply a perspective warp to transform the scene into an overhead view
    * Detect lane lines using a sliding window approach
    * Find best-fit polynomials to represent the lane lines
    * Use the discovered lane lines to calculate lane curvature and vehicle offset
    * Draw the valid lane region and warp it back to the undistorted perspective
    * Overlay the output image with the lane lines and statistics

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


## Camera Calibration

Camera calibration is executed in the lanelines.py ```calibrate_camera``` function.  Using a series of grey-scale converted checkerboard images, all taken from the same camera, checkerboard internal corners are detected using ```cv.findChessboardCorners```.  The found corners, ```imgpoints```, can be aligned with known real-world corner locations, ```objpoints```.  After corresponding points are found in each checkerboard image, I run ```cv2.calibrateCamera``` to determine camera intrinsics, distortion coefficients, and rotation and translation vectors.  The last two are not used in the remainder of the work but are provided so that my function can be used more generally.

With this technique, I am able to undistort the original calibration images such that straight lines in the real world are preserved as straight lines in the image

Distorted
##TODO

Undistorted
##TODO

![alt text][image1]

# Lane Detection Pipeline

This section provides an overview of each step in the pipeline, which is handled a single image at a time

## 1. Undistorting the image

The first step is to remove distortion from the incoming image, which is done by leveraging the camera matrix and distortion coefficients found above.  These, along with the distorted image, are passed to ```cv2.undistort```, which allows us to correct an image from this:

##TODO

to this:
##TODO

## 1. Binary Thresholding

Both color and gradient magnitude thresholding are performed to improve lane detection results.  The former is performed from inside the ```find_lane_lines``` function in ```main.py```.  The undistorted image is converted to the HLS color space then binary thresholding is done on the saturation channel to provide increased resistance to lighting changes.  This produces images such as:

##TODO

Gradient thresholding is done in the ```gradient_threshold()``` function of ```lanelines.py```.  After converting the image to grey-scale, as needed, the gradient is calculated in both x and y directions using the Sobel operator.  The magnitude is computed and the threshold applied generating images like:

##TODO

I combine the two binary images into a single image, adding pixel values directly so that overlapping regions have increased weight.  I decided against including the Direction of Gradient (DoG) threshold as it did not appear to significantly improve performance.  A sample combined threshold image appears below:

##TODO

## Perspective Transformation

Perspective transformation is handled directly in the ```find_lane_lines()``` function in ```main.py```, starting approximately around line 140.  It utilizes OpenCV's ```getPerspectiveTransform()``` and ```warpPerspective()``` functions, using manually selected points to generate the transformation matrix using the former and feeding that, along with the input image, into the latter to generate the transformed image.

The process of selecting the source points (from the ego vehicle view) and destination points (birds-eye view) was entirely manual and required a lot of fine tuning using the ```straight_lines*.jpg``` test images.  I selected the source points to also remove the bit of hood visible in the camera images so that it would not interfere with lane line detection.  I was thereby able to generate top-down perspectives like these:

##TODO

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
