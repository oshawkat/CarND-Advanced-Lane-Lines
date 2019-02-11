import numpy as np
import cv2


def calibrate_camera(fnames, grid_x, grid_y):
    """Calibrate camera using checkerboard images

    This method leverages code from Udacity's Self-Driving Car Nanodegree
    lectures

        Input:
            fnames: file names for the checkerboard images
            grid_x: number of interior checkerboard corners in x (horizontal)
                direction
            grid_y: number of interior checkerboard corners in y (vertical)
                direction

        Output:
            mtx: camera intrinsic matrix
            dist: camera distortion coefficients
            rvecs: rotation vector
            tvecs: translation vector

    """

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((grid_x * grid_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_x, 0:grid_y].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for fname in fnames:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (grid_x, grid_y), None)

        # If found, add object points, image points
        if ret is True:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Calculate camera matrix, distortion, rotation & translation vectors
    img_size = cv2.imread(fnames[0]).shape[1::-1]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                       img_size, None, None)

    return mtx, dist, rvecs, tvecs


def rescale_img(img, max_val=255, min_val=0):
    """Rescale image to within the specified bounds

    Input:
        img: image or numpy array to be scaled
        max_val: greatest element value in output
        min_val: smallest element value in output

    Output:
        Numpy array of type np.uint8 with values ranging
        from as little as min_val up to max_val, inclusive
    """

    output = np.uint8(img / np.max(img) * max_val)
    output = np.maximum(output, np.full_like(output, min_val))

    return output


def gradient_threshold(img, kernel_size, gradmag_min, gradmag_max, dog_min,
                       dog_max):
    """Find image gradients and apply magnitude and orientation thresholding

        Inputs:
            img: source image to calculate gradients
            ker_size: size of sobel gradient kernel
            gradmag_min: minimum gradient magnitude
            gradmag_max: maximum gradient magnitude
            dog_min: minimum orientation direction
            dog_max: maximum orientation direction
        Output:
            gradmag_bin: binary image of gradient magnitude inliers
            dog_bin: binary image of gradient orientation inliers
    """

    # If the input image is not already in greyscale, convert it
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate gradients in x and y directions
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # Calculate the magnitude of the gradient and apply threshold
    sobel_mag = np.sqrt(np.power(sobel_x, 2) + np.power(sobel_y, 2))
    gradmag_bin = binary_threshold(sobel_mag, gradmag_min, gradmag_max)

    # Calculate the direction of the gradient and apply threshold
    sobel_x = np.abs(sobel_x)
    sobel_y = np.abs(sobel_y)
    dog = np.arctan2(sobel_y, sobel_x)
    dog_bin = binary_threshold(dog, dog_min, dog_max)

    return gradmag_bin, dog_bin


def binary_threshold(array, min_val, max_val):
    """Return a binary array of inliers

    Input:
        array: input array to be thresholded against
        min_val: lowest inlier value, inclusive
        max_val: greatest inlier value, inclusive

    Output:
        Returns numpy array of the same size as input
        array.  Inliers set to 1, outliers to 0
    """

    output = np.zeros_like(array)
    output[(array >= min_val) & (array <= max_val)] = 1

    return output


def find_lane_pixels(warped_img, nwindows=9, margin=100, minpix=50):
    """Find lane pixels using a sliding-window approach

    This implementation introduces a technique to leverage actual pixel values
    for the warped_img rather than just using a binary mask input.  This allows
    for pixels that appear in multiple binary thresholds (eg color masking,
    gradient magnitude) to be counted

    WARNING - Passing images with large pixel values (eg >10) as it will result
    in long processing times

    This method leverages code from Udacity's Self-Driving Car Nanodegree
    lectures

        Input:
            warped_img: scene image from top-down view.  Pixel values should
                correspond to confidence that the pixel is a lane line.  Values
                should be kept low (eg < 5)

        Output:
            leftx: x component of pixels in the left lane
            lefty: corresponding y component of pixels in the left lane
            rightx: x component of pixels in the right lane
            righty: corresponding y component of pixels in the right lane
            out_img: input image overlaid with the sliding window boxes found
    """

    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped_img[warped_img.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((warped_img, warped_img, warped_img))
    out_img = rescale_img(out_img)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(warped_img.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero_px = warped_img.nonzero()
    nonzeroy = np.array(nonzero_px[0])
    nonzerox = np.array(nonzero_px[1])
    # Higher quality pixels should count more. Repeat them based on pixel value
    scaled_nonzerox = np.repeat(
        nonzerox, warped_img[nonzero_px].astype(int) + 1, axis=0)
    scaled_nonzeroy = np.repeat(
        nonzeroy, warped_img[nonzero_px].astype(int) + 1, axis=0)
    if nonzerox.shape[0] > scaled_nonzerox.shape[0]:
        print("WARNING - fewer non-zero points after scaling")
    nonzerox = scaled_nonzerox
    nonzeroy = scaled_nonzeroy

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_img.shape[0] - (window + 1) * window_height
        win_y_high = warped_img.shape[0] - window * window_height

        # Find the four below boundaries of the window
        win_xleft_low = max(leftx_current - margin, 0)
        win_xleft_high = min(leftx_current + margin,
                             warped_img.shape[1] - 1)
        win_xright_low = max(rightx_current - margin, 0)
        win_xright_high = min(rightx_current + margin,
                              warped_img.shape[1] - 1)

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If more than the minpix number of pixels are found,
        # recenter the next window
        if len(good_left_inds) > minpix:
            leftx_current = (int)(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = (int)(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped, nwindows=9, margin=100, minpix=50):
    """Determine best-fit polynomials for two lane lines

    Produces 2nd order polynomials that best fit detected lane lines from an
    input image.  Lane lines are found using a sliding-windows technique.This

    This method leverages code from Udacity's Self-Driving Car
    Nanodegree lectures

    Lane line fit is given as [A, B, C] where x = Ay^2 + By + C

        Input:
            binary_warped: binary scene image from top-down view
            nwindows: number of sliding windows
            margin: width of sliding window (x2)
            minpix: minimum number of found pixels to recenter window

        Output:
            left_fit: left lane line best-fit polynomial coefficients
            right_fit: left lane line best-fit polynomial coefficients
            left_fitx: x coordinates of left best-fit line for plotting
            right_fitx: x coordinates of right best-fit line for plotting
            ploty: y coordinates corresponding to left_fitx and right_fitx for
                plotting
            out_img: input image overlaid with sliding window search boxes and
                best-fit lines
    """

    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(
        binary_warped, nwindows, margin, minpix)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = (left_fit[0] * ploty ** 2 + left_fit[1] * ploty +
                     left_fit[2])
        right_fitx = (right_fit[0] * ploty ** 2 + right_fit[1] * ploty +
                      right_fit[2])
    except TypeError:
        # Avoids an error if `left` and `right_fit` are none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    # Visualization
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return left_fit, right_fit, left_fitx, right_fitx, ploty, out_img


def draw_lane_line_area(left_fitx, right_fitx, ploty, img_shape):
    """Draw valid lane region on a blank canvas

        Inputs:
            left_fitx: x coordinates of the left lane line
            right_fitx: x coordinates of the right lane line
            ploty: corresponding y coordinates for both left_fitx and
                right_fitx
            img_shape: tuple of image size (y, x)

        Output:
            numpy array of size [img_shape[0], img_shape[1], 3] with valid
            lane region highlighted in green
    """

    # Create blank canvas of appropriate size
    layer_zero = np.zeros(img_shape[:2], dtype=np.uint8)
    valid_lane_region = np.dstack((layer_zero, layer_zero, layer_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(
        np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the valid lane region onto the warped blank image
    cv2.fillPoly(valid_lane_region, np.int_([pts]), (0, 255, 0))

    return valid_lane_region


def solve_poly_at(poly, eval_pt=0):
    """Solve a 2nd deg polynomial function for a given input

    y = f(x) = Ax^2 + Bx + C

    Input:
        poly: 2nd degree polynomial coefficients in the form [A, B, C]
        eval_pt: x value polynomial should be evaluated at

    Output:
        Value of function evaluated at eval_pt
    """

    return poly[0] * eval_pt ** 2 + poly[1] * eval_pt + poly[2]


def measure_curvature_pixels(poly_fit, y_eval=0):
    '''Calculate curvature of 2nd order polynomial

        Input:
            poly: polynomial on which to calculate curvature
            y_eval: y value along poly at which the curvature
                    is calculated

        Output:
            Radius of curvature, in pixels
    '''

    curverad = ((1 + (2 * poly_fit[0] * y_eval + poly_fit[1]) ** 2) **
                (3 / 2)) / abs(2 * poly_fit[0])

    return curverad


def measure_curvature_meters(poly_fit, y_eval=0, met_per_pix_x=1.,
                             met_per_pix_y=1.):
    '''Calculate curvature of 2nd order polynomial

        Input:
            poly_fit: polynomial on which to calculate curvature
            y_eval: y value along poly at which the curvature
                    is calculated
            met_per_pix_x: scale in x (horizontal) direction as meters/pixel
            met_per_pix_y: scale in y (vertical) direction as meters/pixel

        Output:
            Radius of curvature, in meters
    '''

    # Scale the polynomial coefficients from pixel to meters
    # Per Nanodegree lecture 9.7, if the original polynomial is given as:
    # x = a*(y**2) + b*y + c, the scaled version is:
    # x= mx / (my ** 2) * a* (y**2) + (mx/my)*b*y+c
    poly_fit_meters = np.zeros_like(poly_fit)
    poly_fit_meters[0] = met_per_pix_x / (met_per_pix_y ** 2) * poly_fit[0]
    poly_fit_meters[1] = met_per_pix_x / met_per_pix_y * poly_fit[1]
    poly_fit_meters[2] = poly_fit[2]

    # Using these scaled coefficients, each pixel should correspond to a meter
    curverad_meters = measure_curvature_pixels(poly_fit_meters, y_eval)

    return curverad_meters


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending
    # on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices"
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
