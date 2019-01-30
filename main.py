import os
import glob
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import lanelines as ll


def main():
    print("Starting lane line detection...")

    # Perform camera calibration to enable camera distortion correction
    calib_images = glob.glob('./input/camera_cal/calibration*.jpg')
    grid_size = (9, 6)
    mtx, dist, rvecs, tvecs = ll.calibrate_camera(calib_images, grid_size[0],
                                                  grid_size[1])

    # Undistort and save a few images to validate calibration
    num_pics_to_save = 2
    output_dir = "./output/camera_calibration/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for pic_num in range(num_pics_to_save):
        test_img = cv2.imread(calib_images[pic_num])
        test_undistort = cv2.undistort(test_img, mtx, dist, None, mtx)
        cv2.imwrite(output_dir + "distorted_" + str(pic_num) + ".jpg",
                    test_img)
        cv2.imwrite(output_dir + "undistorted_" + str(pic_num) + ".jpg",
                    test_undistort)

    # Make a list of test images
    images = glob.glob('./input/project_vid/*.png')
    images = glob.glob('./input/project_vid/*22.png')

    # Create output directory to save generated images
    output_dir = "./output/project_vid/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run lane line detection on each of the test images
    for img_path in images:
        # Load image and undistort it using calibrated camera values
        img = cv2.imread(img_path)
        img_name = os.path.basename(img_path)

        undistorted = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imwrite(output_dir + "undistorted_" + img_name,
                    undistorted)
        print(output_dir + "undistorted_" + img_name)

        find_lane_lines(undistorted, img_name, output_dir)

    # Perform lane line detection on a video
    vid_input = "./input/project_video.mp4"
    vid_output = "./output/project_video.mp4"
    # vid_clip = VideoFileClip(vid_input).subclip(0, 5)
    vid_clip = VideoFileClip(vid_input)
    processed_clip = vid_clip.fl_image(
        lambda distorted: vid_process_image(distorted, mtx, dist))
    # processed_clip = vid_clip.fl_image(find_lane_lines)
    processed_clip.write_videofile(vid_output, audio=False)


def vid_process_image(img, cam_mtx, distortion):
    """ Callback function for finding lanelines in video images

        Input:
            img: distorted image to be processed
            cam_mtx: camera calibration matrix for undistorting the image
            distortion: camera distortion coefficients for undistorting the
                image

        Output:
            An image with lane lines and relevant statistics (eg lane
            curvature, lane offset) overlaid

    """

    # Undistort the image then pass it to the lane line detection pipeline
    undistorted = cv2.undistort(img, cam_mtx, distortion, None, cam_mtx)

    return find_lane_lines(undistorted)


def find_lane_lines(undist_img, img_name=None, output_dir=None):
    """Detect best-fit lane lines from an undistorted image

    Process image to detect lane lines then overlay input image with lane
    statistics (curvature, camera offset in lane) and valid lane region

    To enable the saving of debugging images, provide img_name and output_dir

        Input:
            undist_img: Undistorted color image containing lane lines
            img_name: postfix to append to debugging image outputs
            output_dir: path to place debugging images

        Output:
            Color image with valid lane region highlighted in green and lane
            curvature and vehicle lane offset overlaid

    """

    # Apply Gaussian smoothing
    kernel_size = (3, 3)
    smooth = cv2.GaussianBlur(undist_img, ksize=kernel_size, sigmaX=0)

    # Apply color thresholding in the HLS color space
    color_min = 90
    color_max = 255
    hls = cv2.cvtColor(smooth, cv2.COLOR_BGR2HLS)
    sat_channel = hls[:, :, 2]
    color_threshed = ll.binary_threshold(sat_channel, color_min, color_max)

    # Apply gradient magnitude and orientation thresholding
    grad_ker_size = 3
    gradmag_min = 80     # gradient magnitude threshold parameters
    gradmag_max = 200
    dog_min = 1.2     # gradient orientation threshold parameters
    dog_max = 1.4
    grey = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    gradmag_img, dog_img = ll.gradient_threshold(grey, grad_ker_size,
                                                 gradmag_min, gradmag_max,
                                                 dog_min, dog_max)

    # Combine various thresholds into a single mask for use in lane finding
    combined_threshold = ll.rescale_img(gradmag_img + color_threshed)

    # Apply an RoI filter to remove unlikely parts of the image
    (height, width) = combined_threshold.shape
    roi_vertices = np.array([[(0.10 * width, height - 1),     # bottom left
                              (0.40 * width, 0.60 * height),  # top left
                              (0.60 * width, 0.60 * height),  # top right
                              (0.95 * width, height - 1)]],   # bottom right
                            dtype=np.int32)
    mask_unscaled = ll.region_of_interest(combined_threshold, roi_vertices)
    mask = ll.rescale_img(mask_unscaled)

    # Change the lane perspective to a top-down view using points manually
    # selected from a straight-road image
    car_hood = 35  # vertical pixels taken up by the car hood in warped img
    src_pts = np.float32([[698, 450],  # start at top right --> clockwise
                          [1125, mask.shape[0] - car_hood],
                          [210, mask.shape[0] - car_hood],
                          [590, 450]])
    dst_pts = np.float32([[980, 0],
                          [980, mask.shape[0]],
                          [300, mask.shape[0]],
                          [300, 0]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    warped = cv2.warpPerspective(mask, M, (mask.shape[1], mask.shape[0]))

    # Find lane lines using sliding windows technique
    nwindows = 20    # Choose the number of sliding windows
    margin = 50    # Set the width of the windows +/- margin
    minpix = 50     # Set minimum number of pixels found to recenter window
    leftx, lefty, rightx, righty, sliding_img = ll.find_lane_pixels(
        warped, nwindows, margin, minpix)
    l_fit, r_fit, l_fitx, r_fitx, ploty, poly_img = ll.fit_polynomial(
        warped, nwindows, margin, minpix)

    # Find lane curvature
    met_per_pix_x = 3.7 / 580
    met_per_pix_y = 30 / 720
    # left_curvature = ll.measure_curvature_pixels(l_fit, warped.shape[0])
    # right_curvature = ll.measure_curvature_pixels(r_fit, warped.shape[0])
    left_curvature_met = ll.measure_curvature_meters(
        l_fit, warped.shape[0], met_per_pix_x, met_per_pix_y)
    right_curvature_met = ll.measure_curvature_meters(
        r_fit, warped.shape[0], met_per_pix_x, met_per_pix_y)
    curvature_met = (left_curvature_met + right_curvature_met) / 2.0

    # Find vehicle offset in lane
    # This variable is the x position of the middle of the lane
    # corresponding to being centered in the lane.  Measured manually from
    # straight_lines2 example image
    zero_offset = 622
    left_pos = ll.solve_poly_at(l_fit, warped.shape[0] - car_hood)
    right_pos = ll.solve_poly_at(r_fit, warped.shape[0] - car_hood)
    lane_midpoint = (left_pos + right_pos) / 2.0
    lane_offset_px = lane_midpoint - zero_offset
    lane_offset_m = lane_offset_px * met_per_pix_x

    # Draw lane lines on original input image by first generating a valid
    # ll region in the top view then warping it to the camera perspective
    # and overlaying it on the original undistorted image
    top_ll_area = ll.draw_lane_line_area(l_fitx, r_fitx, ploty,
                                         undist_img.shape)
    warped_ll_area = cv2.warpPerspective(
        top_ll_area, M_inv, (undist_img.shape[1], undist_img.shape[0]))
    ll_overlay = cv2.addWeighted(undist_img, 1, warped_ll_area, 0.3, 0)

    # Overlay image with lane curvature and vehicle lane offset
    text_curv_position = (30, 60)
    text_offset_position = (30, 130)
    text_color = (255, 255, 255)
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = 2
    text_thickness = 2
    cv2.putText(ll_overlay, "Radius of Curvature (m): " +
                str(int(curvature_met)), text_curv_position, text_font,
                text_size, text_color, text_thickness)
    if lane_offset_m < 0:
        cv2.putText(ll_overlay, "Vehicle is " +
                    str(abs(np.round(lane_offset_m, 2))) + "m left of center",
                    text_offset_position, text_font,
                    text_size, text_color, text_thickness)
    else:
        cv2.putText(ll_overlay, "Vehicle is " +
                    str(abs(np.round(lane_offset_m, 2))) + "m right of center",
                    text_offset_position, text_font,
                    text_size, text_color, text_thickness)

    # Save debugging images if enabled (by providing image name and output dir)
    if img_name is not None and output_dir is not None:
        # Save images for various steps in the pipeline
        cv2.imwrite(output_dir + "smooth_" + img_name, smooth)
        cv2.imwrite(output_dir + "sat_channel_" + img_name,
                    sat_channel)
        cv2.imwrite(output_dir + "color_threshed_" + img_name,
                    ll.rescale_img(color_threshed))
        cv2.imwrite(output_dir + "grey_" + img_name, grey)
        cv2.imwrite(output_dir + "gradmag_" + img_name,
                    ll.rescale_img(gradmag_img))
        cv2.imwrite(output_dir + "dog_" + img_name,
                    ll.rescale_img(dog_img))
        cv2.imwrite(output_dir + "thresholds_" + img_name,
                    mask)
        cv2.imwrite(output_dir + "warped_" + img_name,
                    ll.rescale_img(warped))
        cv2.imwrite(output_dir + "sliding_win_" + img_name,
                    sliding_img)
        cv2.imwrite(output_dir + "ll_overlay_" + img_name, ll_overlay)
        # Plots the left and right best-fit polynomials on the lane lines
        plt.imshow(poly_img)
        plt.plot(l_fitx, ploty, color='yellow')
        plt.plot(r_fitx, ploty, color='yellow')
        plt.xlim(0, poly_img.shape[1])
        plt.ylim(poly_img.shape[0], 0)
        plt.savefig(output_dir + "poly_" + img_name)
        plt.clf()
        # Display lane curvature and offset
        print("Lane pixel to meter scale values are incorrect.  Please update")
        print("Left lane curvature:  ", int(round(left_curvature_met)))
        print("Right lane curvature: ", int(round(right_curvature_met)))
        print("Lane is offset in px: ", round(lane_offset_px, 1))
        print("Lane is offset in m:  ", round(lane_offset_m, 3))

    return ll_overlay


if __name__ == "__main__":
    main()
