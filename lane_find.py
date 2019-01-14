#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 20:37:10 2019

@author: mingrui
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os


from sklearn.linear_model import RANSACRegressor

#%matplotlib inline

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)



####################################
# My code starting from here
###################################

# all the parameters needed in the whole 
# lane detection pipeline
class LaneDetectParam:
    def __init__(self):
        # for color detection
        self.white_rgb_threshold = np.array([90, 90, 90])
        self.yellow_hsv_lb = np.array([15, 100, 100])
        self.yellow_hsv_ub = np.array([40, 255, 255])
        
        # for interest region
        self.ir_row_ratio = 0.4
        self.ir_upper_col_ratio = 0.05
        self.ir_lower_col_ratio = 1.0
        
        # for Gaussian smoothing
        self.kernel_size = 5
        
        # for canny edge detection
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
      
        # for hough tranform
        self.rho = 1
        self.theta = np.pi/180
        self.min_vote = 5
        self.min_line_len = 5
        self.max_line_gap = 1
        
        # for line drawing
        self.line_color = [255, 0, 0]
        self.line_thickness = 10
        
# detect white areas
def detect_white(image, param):
    mask = (image[:,:,0] > param.white_rgb_threshold[0]) \
         & (image[:,:,1] > param.white_rgb_threshold[1]) \
         & (image[:,:,2] > param.white_rgb_threshold[2]) \
    
    return mask

# detect yellow areas
def detect_yellow(image, param):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, param.yellow_hsv_lb, param.yellow_hsv_ub)
    
    return mask > 0
    
# mask of interest region based on colors
def get_color_mask(image, param):
    mask_white = detect_white(image, param)
    mask_yellow = detect_yellow(image, param)
    mask = mask_white | mask_yellow
    
    return mask

# vertices of the interest region
def get_interest_region_vertices(image, param):
    row, col = image.shape[:2]
    mid = col//2
 
    # height of the interest region
    height = int(row * param.ir_row_ratio)
    
    # length of the upper and lower horizontal edges
    upper_edge_length = int(col * param.ir_upper_col_ratio)
    lower_edge_length = int(col * param.ir_lower_col_ratio)
    offset = (col - lower_edge_length) // 2
    
    # trapezoid
    vertices = np.array([[(offset, row-1), (col-1-offset, row-1), \
                          (mid+upper_edge_length//2, row-height), (mid-upper_edge_length//2, row-height)]])
    return vertices

# mask of interest region based on positions
def get_region_mask(image, param):
    vertices = get_interest_region_vertices(image, param)
    
    if len(image.shape) > 2:
        mask = np.zeros_like(image[:, :, 0])
    else:
        mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, vertices, 1) > 0
   
    return mask

def get_mask(image, param):
    region_mask = get_region_mask(image, param)
    color_mask = get_color_mask(image, param)   

    mask = region_mask & color_mask
    return mask


# group all the line segments into two groups: left and right
def get_left_right_lines(lines):
    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1 == x2:
            continue
        param = np.polyfit((x1, x2), (y1, y2), 1)
   
        slope = param[0]
        if slope < 0:
            left_lines.append(line)
        elif slope > 0:
            right_lines.append(line)
            
    return np.array(left_lines), np.array(right_lines)
               
# fit one line based on the input line segments
def fit_one_line(lines):
    X = []
    Y = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        X.append(x1)
        X.append(x2)
        Y.append(y1)
        Y.append(y2)
        
    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y).reshape(-1, 1)
    ransac = RANSACRegressor()
    model = ransac.fit(X, Y)
    
    return np.array([model.estimator_.coef_, model.estimator_.intercept_])
        
# draw the given line into the given image
def draw_one_line(image, line, param):
    slope, intercept = line
    row, col = image.shape[:2]
    
    # y coordinates 
    y1 = row-1
    y2 = row - int(row * param.ir_row_ratio)
    
    # compute x coordinates
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    
    # having obtained to points, draw the line
    cv2.line(image, (x1, y1), (x2, y2), param.line_color, param.line_thickness)


# given detected houg lines, draw full extent lane lines
def draw_full_lanes(line_image, lines, param):
    # group all the line segments into two groups: left and right
    left_lines, right_lines = get_left_right_lines(lines)
    
    # fit one line for each group
    left_line = fit_one_line(left_lines)
    right_line = fit_one_line(right_lines)
    
    # draw the lane lines
    draw_one_line(line_image, left_line, param)
    draw_one_line(line_image, right_line, param)
    
    return line_image
   
# given the output of a Canny transform, 
# return an image with hough lines drawn
def hough_lines(edge_image, param):
    lines = cv2.HoughLinesP(edge_image, param.rho, param.theta, param.min_vote, np.array([]), param.min_line_len, param.max_line_gap)
    line_img = np.zeros((edge_image.shape[0], edge_image.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, param.line_color, param.line_thickness)
    
    return line_img

# given the output of a Canny transform, 
# return an image with full extent lanes
def hough_full_lines(edge_image, param):
    lines = cv2.HoughLinesP(edge_image, param.rho, param.theta, param.min_vote, np.array([]), param.min_line_len, param.max_line_gap)
    line_image = np.zeros((edge_image.shape[0], edge_image.shape[1], 3), dtype=np.uint8)
    line_image = draw_full_lanes(line_image, lines, param)

    return line_image


# lane find pipeline
# mode 0: find lane line segments
# mode 1: find full extent of the lanes
def lane_find(image, param, mode):
    gray_img = grayscale(image)
    
    # edge detection
    blurred_img = gaussian_blur(gray_img, param.kernel_size)
    edge_image = canny(blurred_img, param.canny_low_threshold, param.canny_high_threshold)
    
    # masked edge image
    mask = get_mask(image, param)
    masked_edge_image = edge_image & mask
    
    # find lane lines
    if mode == 0:
        line_image = hough_lines(masked_edge_image, param)
    else:
        line_image = hough_full_lines(masked_edge_image, param)
    
    # combine with the input image
    with_line_image = weighted_img(line_image, image)
    
    return with_line_image, line_image, edge_image, masked_edge_image

from moviepy.editor import VideoFileClip
from IPython.display import HTML
def save_frames():
    T1 = 0
    T2 = 4
    N = 40
    clip1 = VideoFileClip("test_videos/challenge.mp4").subclip(T1, T2)
    for t in range(N):
        clip1.save_frame('tmpvideos/frame_' + str(t) + '.jpg', t=(T2-T1)*t/N)


# test lane find
def test_lane_find(input_path, output_path, mode=1, show_image=False, debug_image_name=''):
    # parameter
    param = LaneDetectParam()
    
    # make sure the output path exists
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    # detect lanes in the images in the input path
    image_list = os.listdir(input_path)
    for image_name in image_list:
     
        # show_image=True means just debug for the given image
        if show_image and image_name != debug_image_name:
           continue
         
        print(image_name)
        I = mpimg.imread(os.path.join(input_path, image_name))
        with_line_image, line_image, edge_image, masked_edge_image = lane_find(I, param, mode)
        
        if show_image:
            plt.imshow(I)
            plt.show()

            plt.imshow(with_line_image)
            plt.show()
            
            plt.imshow(line_image)
            plt.show()
            
            plt.imshow(masked_edge_image)
            plt.show()
        else:
            mpimg.imsave(os.path.join(output_path, image_name), with_line_image)
        
        
        
        
        
if __name__ == '__main__':
    #test_lane_find('test_images', 'test_images_output', mode=0, show_image=False, debug_image_name='solidWhiteRight.jpg')
    #save_frames()
    test_lane_find('frames', 'frames_output', mode=1, show_image=False, debug_image_name='frame_0.jpg')
    