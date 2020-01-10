import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from calibration_utils import calibrate_camera, undistort
from binariation_utils import binarize

def birdeye(img, verbose= False):

    h, w = img.shape[:2]

    src = np.float32([[w, h-10],
                      [0, h-10],
                      [546, 460],
                      [732, 460]])

    dst = np.float32([[w, h],
                      [0, h],
                      [0, 0],
                      [w, 0]])

    M = cv2.getPrespectiveTransform(src, dst)
    Minv = cv2.getPrespectiveTransform(dst, src)

    warped = cv2.warpPrespective(img, M, (w,h), flags= cv2.INTER_LINEAR)

    if verbose:
        f, axarray = plt.subplots(1, 2)
        f.set_facecolor('white')
        axarray[0].set_title('Before prespective transform')
        axarray[0].imshow(img, cmap= 'gray')
        for point in src:
            axarray[0].plot(*point, '.')
        axarray[1].set_title('After prespective transform')
        axarray[1].imshow(warped, cmap= 'gray')
        for point in dst:
            axarray[1].plot(*point, '.')
        for axis in axarray:
            axis.set_axis_off()
        plt.show()

    return wraped, M, Minv

if __name__ == '__main__':

    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')

    for test_img in glob.glob('test_images/*.jpg'):
        img = cv2.imread(test_img)
        img_undistorted = undistort(img, mtx, dist, verbose= False)
        img_binary = binarize(image_undistorted, verbose= False)

        img_birdeye, M, Minv = birdeye(cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB))
        
