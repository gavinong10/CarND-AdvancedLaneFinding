import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import functools

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, thresh_min=0, thresh_max=255):
        # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh_min) & (gradmag <= thresh_max)] = 1

    # Return the binary image
    return binary_output

def dir_threshold(image, sobel_kernel=3, thresh_min=0, thresh_max=np.pi/2): #thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh_min) & (absgraddir <= thresh_max)] = 1

    # Return the binary image
    return binary_output

def extract_s_channel(img, thresh_min = 30, thresh_max = 255, repeat=False):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh_min) & (s_channel <= thresh_max)] = 1
    
    if repeat:
        return np.repeat(s_binary[:,:,np.newaxis], 3, axis=2)
    return s_binary


def pipeline(img, thresh_min = None, thresh_max = None):
    pipeline_res = [
        abs_sobel_thresh(img, thresh_min=20, thresh_max=85, orient='x'),
        abs_sobel_thresh(img, thresh_min=50, thresh_max=120, orient='y'),
        mag_thresh(img, thresh_min=60, thresh_max=120, sobel_kernel=3),
        #dir_threshold(img, thresh_min=12 * np.pi/2/20, thresh_max=13 * np.pi/2/20, sobel_kernel=3),
        extract_s_channel(img, thresh_min=150, thresh_max=255, repeat=False)
    ]
    
    return functools.reduce(np.logical_or, pipeline_res)

def getPerspectiveTransformMatrices(sample_img):
    img_size = (sample_img.shape[1], sample_img.shape[0])
   
    src = np.float32(
        [[250, 680],
        [590, 450],
        [690, 450],
        [1055, 680]])
    
    dst = np.float32(
        [[320, 680],
        [320, -700],
        [955, -700],
        [955, 680]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    Mi = cv2.getPerspectiveTransform(dst, src)
    
    return M, Mi

def warp(img, M=None):
    if M is None:
        M, _ = getPerspectiveTransformMatrices(img)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

if __name__ == "__main__":
    pass
    # combined = np.zeros_like(dir_binary)
    # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1