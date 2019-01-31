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

[distorted_checker]: ./output/camera_calibration/distorted_.jpg
[undistorted_checker]: ./output/camera_calibration/undistorted_.jpg
[distorted]: ./input/test_images/test1.jpg
[undistorted]: ./output/test_images/undistorted_test1.jpg
[color_thresh]: ./output/test_images/color_threshed_test4.jpg
[grad_mag_thresh]: ./output/test_images/gradmag_test4.jpg
[combined_thresh]: ./output/test_images/thresholds_test4.jpg
[warped]: ./output/test_images/warped_test4.jpg
[best_fit]: ./output/test_images/poly_test2.jpg
[ll_overlay]: ./output/test_images/ll_overlay_test2.jpg


## Camera Calibration

Camera calibration is executed in the lanelines.py ```calibrate_camera``` function.  Using a series of grey-scale converted checkerboard images, all taken from the same camera, checkerboard internal corners are detected using ```cv.findChessboardCorners```.  The found corners, ```imgpoints```, can be aligned with known real-world corner locations, ```objpoints```.  After corresponding points are found in each checkerboard image, I run ```cv2.calibrateCamera``` to determine camera intrinsics, distortion coefficients, and rotation and translation vectors.  The last two are not used in the remainder of the work but are provided so that my function can be used more generally.

With this technique, I am able to undistort the original calibration images such that straight lines in the real world are preserved as straight lines in the image

Distorted:

![Original distorted checkerboard][distorted_checker]

Undistorted:

![Undistorted checkerboard][undistorted_checker]

# Lane Detection Pipeline

This section provides an overview of each step in the pipeline, which is handled a single image at a time

## 1. Undistorting the image

The first step is to remove distortion from the incoming image, which is done by leveraging the camera matrix and distortion coefficients found above.  These, along with the distorted image, are passed to ```cv2.undistort```, which allows us to correct an image from this:

![Distorted lane view][distorted]

to this:

![Undistorted lane view][undistorted]

## 1. Binary Thresholding

Both color and gradient magnitude thresholding are performed to improve lane detection results.  The former is performed from inside the ```find_lane_lines``` function in ```main.py```.  The undistorted image is converted to the HLS color space then binary thresholding is done on the saturation channel to provide increased resistance to lighting changes.  This produces images such as:

![Color Thresholded image][color_thresh]

Gradient thresholding is done in the ```gradient_threshold()``` function of ```lanelines.py```.  After converting the image to grey-scale, as needed, the gradient is calculated in both x and y directions using the Sobel operator.  The magnitude is computed and the threshold applied generating images like:

![Gradient Magnitude thresholded image][grad_mag_thresh]

I combine the two binary images into a single image, adding pixel values directly so that overlapping regions have increased weight.  I decided against including the Direction of Gradient (DoG) threshold as it did not appear to significantly improve performance.  I also applied an ROI filter to remove superfluous portions of the scene.  A sample combined threshold image appears below:

![Joint threshold image][combined_thresh]

## 1. Perspective Transformation

Perspective transformation is handled directly in the ```find_lane_lines()``` function in ```main.py```, starting approximately around line 140.  It utilizes OpenCV's ```getPerspectiveTransform()``` and ```warpPerspective()``` functions, using manually selected points to generate the transformation matrix using the former and feeding that, along with the input image, into the latter to generate the transformed image.

The process of selecting the source points (from the ego vehicle view) and destination points (birds-eye view) was entirely manual and required a lot of fine tuning using the ```straight_lines*.jpg``` test images.  I selected the source points to also remove the bit of hood visible in the camera images so that it would not interfere with lane line detection.  I was thereby able to generate top-down perspectives like these:

![Perspective transformation into the birds-eye view][warped]

## 1. Lane Line Detection

All the prior steps are precursors for the main event - lane line detection.  The core of this functionality lies in the ```fit_polynomial()``` function in the ```lanelines.py``` module.  The first step is to call the ```find_lane_pixels()``` method that, like the example demonstrated in the lecture, uses a sliding window technique on the perspective transformed binary image.  The algorithm first selects likely starting points for each lane by creating a histogram of pixel values in the lower half of the image along the vertical direction.  The maximal columns of both horizontal halves of the histogram seed the initial positions of their respective lanes.

The image is broken into horizontal strips of a defined number (function parameter) and, for each window, starting from the bottom of the image, the activated pixels within a given width of likely x position are counted.  If there are a sufficient number, starting window position can shift toward the higher density.  Finally, this sub-function returns the coordinates of these 'found' lane line pixels

Then, I use ```np.polyfit()``` to calculate a 2nd order polynomial that best fits the found pixels.  The output of this portion of the pipeline looks like this:

![Best-fit lane lines using the sliding window method][best_fit]

## 1. Lane Curvature and Offset

Lane curvature is calculated in the ```measure_curvature_meters()``` function in the ```lanelines.py``` module.  I first adjust the best-fit line coefficients to scale them from pixel space to meters (the exact values of this conversion will depend on your original scene and perspective warp).  Then I calculated the curvature formula we derived in lecture.  This resulted in two curvature values, one for each lane, which I averaged.

Calculating lane offset took some manual pixel measurements.  I calculated the offset of the vehicle in one of the ``straight_lines*.jpg``` test images by measuring distance to each lane line from the image center and calculating that as a meter measure given assumptions about lane width.  With that as my baseline (as I had also used the image to find perspective warp values), I now had a reference point with which to compare all subsequent images.  By calculating the intersections of the best-fit lines and the base of the warped image, and a pixel to meters scale, I determine lane offset.  This is written in ```main.py```, around line 180.

Both lane curvature and offset statistics are overlaid on the final output images

## 1. Final Output

The valid lane region is drawn in the warped perspective then the inverse-warp transformation is applied to turn it back into the ego vehicle view.  Then this and lane statistics are overlaid on the undistorted image as output, which look like:

![Annotated output image with valid lane region and lane line statistics][ll_overlay]

You can view a processed video scene [here](./output/project_video.mp4) as well

---

# Discussion

This lane finding methodology, though significantly more robust than the prior Canny edges based approach, is not without its shortcomings.  Significant parameter tuning is required, so it may not be entirely generalizable to all roads and lighting conditions.  It is prone to failure around:
    * Large shadows or lighting changes in the lane
    * Cracks or changes in pavement color within the lane
    * Nearby vehicles in adjacent lanes
    * Highly curved lanes
    * Hills or lane occlusions

The largest improvement in my lane finding is likely to be in adding some type of persistence between images in a video.  That way, when lane lines are difficult to detect or incorrectly detected, I can use prior results for a few frames instead.