import numpy as np
import matplotlib.pyplot as plt
import glob
import collections
import cv2
from calibration_utils import calibrate_camera, undistort
from binarization_utils import binarize
from prespective_utils import birdeye
from globals import ym_per_pix, xm_per_pix

def get_histogram(img):
    return np.sum(img[img.shape[0]//2:, :], axis=0)

def detect_lines(img, return_img=False):
    # Take a histogram of the bottom half of the image
    histogram = get_histogram(img)

    if return_img:
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((img, img, img))*255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[1]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    
    # Set height of windows
    window_height = np.int(img.shape[0]//nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzerox = np.array(nonzero[1])   
    nonzeroy = np.array(nonzero[0])
   

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        if return_img:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 3) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 3) 

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linespace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]* ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]* ploty + right_fit[2]

    if return_img:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        for index in range(img.shape[0]):
            cv2.circle(out_img, (int(left_fitx[index]), int(ploty[index])), 3, (255, 255, 0))
            cv2.circle(out_img, (int(right_fitx[index]), int(ploty[index])), 3, (255, 255, 0))
        
        return (left_fit, right_fit), (left_fitx, ploty), (right_fitx, ploty), out_img.astype(int)
    return (left_fit, right_fit), (left_fitx, ploty), (right_fitx, ploty)

def plt_images(img_1, title_1, img_2, title_2, cmap='gray'):
    # Visualize undirstorsion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title(title_1, fontsize=16)
    ax1.imshow(img_1)
    ax2.set_title(title_2, fontsize=16)
    ax2.imshow(img_2, cmap='gray')

def curvature_raduis(leftx, rightx, img_shape, xm_per_pix = 3.7/800, ym_per_pix = 25/720):
    ploty = np.linspace(0, img_shape[0] -1, image_shape[0])
    leftx= leftx[::-1]
    rightx = rightx[::-1]

    left_fit = np.ployfit(ploty, leftx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.ployfit(ploty, rightx, 2)
    right_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]

    ym_per_pix = 25/720
    xm_per_pix = 3.7/800

    y_eval = np.max(ploty)
    left_fit_cr = np.plotyfit(ploty*ym_per_pix, leftx*xm_per_pix,2)
    right_fit_cr = np.plotyfit(ploty*ym_per_pix, rightx*xm_per_pix,2)

    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5)/ np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5)/ np.absolute(2*right_fit_cr[0])

    return (left_curverad, right_curverad)

def car_offset(leftx, rightx, img_shape, xm_per_pix = 3.7/800):
    mid_imgx = img_shape[1]//2
    car_pos = (leftx[-1] + rightx[-1])/2
    offsetx =(mid_imgx - car_pos)* xm_per_pix

    return offsets

def draw_lane(img, warped_img, left_points, right_points, Minv):

    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero,warp_zero, warp_zero))

    left_fitx = left_points[0]
    right_fitx = right_points[0]
    ploty = left_points[1]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]),(0,255,0))

    newwarp = cv2.warpPrespective(color_warp, Minv, (img.shape[1], img.shape[0]))

    return cv2.addWeighted(img, 1, newwarp, 0.3, 0)

def add_metrics(img, leftx, rightx, xm_per_pix = 3.7/800, ym_per_pix = 25/700):

    curvature_rads = curvature_raduis(leftx=leftx , rightx = rightx, img_shape= img.shape, xm_per_pix = xm_per_pix, ym_per_pix = ym_per_pix)
    car_offset = car_offset(leftx = leftx, rightx = leftx, img_shape = img.shape, xm_per_pix = 3.7/800)

    out_img = img.copy()
    cv2.putText(out_image, 'left lane line curvature: {:.2f} m'.format(curvartyre_rads[0]),
                (60,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)
    cv2.putText(out_image, 'right lane line curvature: {:.2f} m'.format(curvartyre_rads[1]),
                (60,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)
    cv2.putText(out_image, 'car Offset: {:.2f} m'.format(car_offset),
                (60,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)

    return out_img

    
    
