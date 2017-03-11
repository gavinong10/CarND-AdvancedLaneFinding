from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
from findlines import *
from least_squares_circle import *
from transforms import *
import cv2

# Implement sanity checks to reject unusable results and replace them with a result from prior frames. Some form of averaging over a few frames may be helpful. However don't over do it, because it is important to avoid reacting too slowly on curves or to changes in vehicle position within the lane.
# Continue to investigate color spaces to find a better thresholding solution. The goal is to rely more on color identification and less on gradients (which are not so useful in shadows or changing road conditions). The R and V color channels are strongly represented in yellows and whites.
# Capture images of video frames where problems are occurring and run the pipeline on those images to avoid long processing times while trying to solve a localized problem.
# Use an area around the lines fitted in prior images to search for lane pixels in a new image. But be prepared to do a full histogram search if valid data is not obtained for a few frames.

def output_video(pipeline, input_video="project_video.mp4", output_video="project_video_out.mp4"):
    input_clip = VideoFileClip(input_video)
    write_clip = input_clip.fl_image(pipeline) #NOTE: Color images expected
    write_clip.write_videofile(output_video, audio=False)
    
def get_sample_clip(input_video="project_video.mp4"):
    vidcap = cv2.VideoCapture(input_video)
    success, image = vidcap.read()
    return image
    
def process_thresholding(warped, **kwargs):
    # Experimental version
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gradx = abs_sobel_thresh(warped, orient='x', thresh_min=5, thresh_max=90)
    grady = abs_sobel_thresh(warped, orient='y', thresh_min=50, thresh_max=120)
    mag_binary = mag_thresh(warped, sobel_kernel=3, thresh_min=60, thresh_max=120)
    #dir_binary = dir_threshold(warped, sobel_kernel=3, thresh_min=0.7, thresh_max=1.3)
    hls_binary = extract_s_channel(warped, thresh_min=150, thresh_max=250)
    #combined = np.zeros_like(dir_binary)
    combined = np.zeros_like(hls_binary)
    combined[(gradx == 1) | (hls_binary == 1) | (mag_binary == 1)] = 1
    #combined[((gradx == 1) & (hls_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    # # & version
    # gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # gradx = abs_sobel_thresh(warped, orient='x', thresh_min=10, thresh_max=230)
    # grady = abs_sobel_thresh(warped, orient='y', thresh_min=10, thresh_max=230)
    # mag_binary = mag_thresh(warped, sobel_kernel=3, thresh_min=30, thresh_max=150)
    # dir_binary = dir_threshold(warped, sobel_kernel=3, thresh_min=0.7, thresh_max=1.3)
    # hls_binary = extract_s_channel(warped, thresh_min=80, thresh_max=255)
    # combined = np.zeros_like(dir_binary)
    # combined[((gradx == 1) & (hls_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    return combined

def radius_calcs(yvals, left_fit, right_fit):
    ################
    # Pick 3 points to evaluate the x values given 3 y values and the polynomial coefficients
    y_eval1 = np.max(yvals)
    y_eval2 = np.mean(yvals)
    y_eval3 = np.min(yvals)
    left_fitx_1 = left_fit[0]*y_eval1**2 + left_fit[1]*y_eval1 + left_fit[2]
    left_fitx_2 = left_fit[0]*y_eval2**2 + left_fit[1]*y_eval2 + left_fit[2]
    left_fitx_3 = left_fit[0]*y_eval3**2 + left_fit[1]*y_eval3 + left_fit[2]
    right_fitx_1 = right_fit[0]*y_eval1**2 + right_fit[1]*y_eval1 + right_fit[2]
    right_fitx_2 = right_fit[0]*y_eval2**2 + right_fit[1]*y_eval2 + right_fit[2]
    right_fitx_3 = right_fit[0]*y_eval3**2 + right_fit[1]*y_eval3 + right_fit[2]

    # Calculated the turning center point xc, yc and radius (currently just using 3 points): 

    #lm1, lm2, lxc, lyc, lradius = find_3p_circle_radius(left_fitx_1,y_eval1,left_fitx_2,y_eval2,left_fitx_3,y_eval3,)
    lxc, lyc, lradius, _ = leastsq_circle(np.array([left_fitx_1, left_fitx_2, left_fitx_3]), np.array([y_eval1, y_eval2, y_eval3]))
    l_steering_angle = 5*360/lxc # assume xc <> 0, xc and radius value is very close, xc will show the direction as well


    #rm1, rm2, rxc, ryc, rradius = find_3p_circle_radius(right_fitx_1,y_eval1,right_fitx_2,y_eval2,right_fitx_3,y_eval3,)
    rxc, ryc, rradius, _ = leastsq_circle(np.array([right_fitx_1, right_fitx_2, right_fitx_3]), np.array([y_eval1, y_eval2, y_eval3]))

    #r_steering_angle = 5*360/rxc # assume xc <> 0, xc and radius value is very close, xc will show the direction as well
    #steering_angle = l_steering_angle + r_steering_angle
    turning_radius = (lradius+rradius)/2 # smooth out the radius

    return turning_radius

def eval_poly_on_points(fit, ypoints):
    result = 0
    degree = len(fit)
    for idx in range(degree):
        result += fit[idx] * ypoints ** (degree - idx - 1)
    return result
    
# Have a look at OpenCV's matchShapes function. It compares two shapes and returns an indication of how similar they are. Using it to compare the current polygon to one from a prior frame can provide a method for rejecting a bad frame by allowing the pipeline to use the last good polygon instead.

def width_checker(leftx_base, rightx_base):
    pixel_lane_width = 620
    return abs(rightx_base-leftx_base-pixel_lane_width) < 0.2 * pixel_lane_width
    
def return_pipeline(mtx, dist, M, Mi):
    left_fit, right_fit = None, None
    base_history = [] 
    
    def averager(leftx_base, rightx_base, frames=5):
        nonlocal base_history        
        if len(base_history) == 0:
            if not width_checker(leftx_base, rightx_base):
                return None
            for _ in range(frames):
                base_history.append((leftx_base, rightx_base))
        elif width_checker(leftx_base, rightx_base):
            base_history.append((leftx_base, rightx_base))
            base_history = base_history[1:]
        return [int(np.mean(history)) for history in zip(*base_history)]

        # NOTE: Exponential smoothing is an alternative to averaging over N frames. If you have a New frame and an Old frame, smooth by updating the New as follows: New = gamma New + (1-gamma) Old. 0 < gamma < 1.0
        
    last_img = None
    def check_fit(img, thresh=0.01):
        nonlocal last_img, base_history
        if last_img is None:
            last_img = img
            return img, -1
        
        imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgray_last = cv2.cvtColor(last_img,cv2.COLOR_BGR2GRAY)
        
        contours = cv2.findContours(imgray,2,1)
        cnt1 = contours[0]
        contours = cv2.findContours(imgray_last,2,1)
        cnt2 = contours[0]

        ret = cv2.matchShapes(cnt1,cnt2,1,0.0)

        if ret < thresh:
            base_history[-1] = tuple(int(np.mean(history)) for history in zip(*base_history))
        else:
            last_img = img
            
        return last_img, ret

    def pipeline(img):
        nonlocal left_fit, right_fit #Tracks the last img that was seen
        
        #img = cv2.resize(img, (720, 405))
        img = cv2.undistort(img, mtx, dist, None, mtx)
        #warped = perspective_transform(img, M)
        warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        
        combined = process_thresholding(warped) # A 2 dimensional image of thresholded pixels
        
        # NOTE: Below is a selector for the quick search method based on if a fit had been performed in the last stage.
        # Currently, it is set to True because fit_poly_next needs extra work - if the detections start to get off track,
        # we need to recalibrate fully. Flagged for implementation in future as an improvement.
        
        if True: #left_fit is None and right_fit is None:
            # Create the initial histogram, fit the sliding windows and fit a 2nd order polynomial to them
            histogram = np.sum(combined[int(combined.shape[0]/2):,:], axis=0)
            midpoint, leftx_base, rightx_base = return_bases(histogram)
            # Implement an averager technique on the bases
            averages = averager(leftx_base, rightx_base, 5)
            if averages is None:
                return img # data point used in averager as a calibrator
            leftx_base, rightx_base = averages
            left_fit, right_fit, leftx, lefty, rightx, righty, _ = fit_poly(combined, midpoint, leftx_base, rightx_base)
            
        else:
            # We had already done a blind search before, so now search within margins
            left_fit, right_fit, leftx, lefty, rightx, righty, _ = fit_poly_next(combined, left_fit, right_fit)
        
        # create a variable containing all the pixel values for y axis to aid in evaluation
        yvals = np.linspace(0, img.shape[0], num=img.shape[0])
        
        turning_radius = radius_calcs(yvals, left_fit, right_fit)

        # Find camera position
        left_mean = np.mean(leftx)
        right_mean = np.mean(rightx)
        camera_pos = (combined.shape[1]/2)-np.mean([left_mean, right_mean])

        # Define conversions in x and y from pixels space to meters

        xm_per_pix = 3.7/620 # meteres per pixel in x dimension
        ym_per_pix = xm_per_pix # meters per pixel in y dimension
        
        left_fit_cr = np.polyfit(np.array(lefty,dtype=np.float32)*ym_per_pix, \
                             np.array(leftx,dtype=np.float32)*xm_per_pix, 2)
        right_fit_cr = np.polyfit(np.array(righty,dtype=np.float32)*ym_per_pix, \
                              np.array(rightx,dtype=np.float32)*xm_per_pix, 2)

        # Return radius of curvature is in meters
        y_eval = np.max(yvals)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) \
                                 /np.absolute(2*left_fit_cr[0])

        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) \
                                    /np.absolute(2*right_fit_cr[0])

        # For all the integer pixel values of the image y axis, ascertain the appropriate x value from the polyfit
        left_fitx = eval_poly_on_points(left_fit, yvals)
        right_fitx = eval_poly_on_points(right_fit, yvals)
        # Link all points of the evaluated polynomial for cv2.fillPoly() in pix space
        pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        cv2.fillPoly(warp_zero, np.int_([pts]), (0,255, 0))
        cv2.polylines(warp_zero, np.array([pts_left], dtype=np.int32), False,(255,0,0),thickness = 15)
        cv2.polylines(warp_zero, np.array([pts_right], dtype=np.int32), False,(0,0,255),thickness = 15)
        
        # Checks if warp_zero is within acceptable tolerance of the last image and if not, returns last image AND
        # modifies the base history's last value to be the mean of the full history
        
        warp_zero, ret = check_fit(warp_zero)

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(img,'Camera Position' + ' [' + str(camera_pos*xm_per_pix)[:6] + '] m',(10,30), font, 1,(255,255,255),2)
        cv2.putText(img,'Turning Radius ' +str(turning_radius)[:6] + '] m' ,(10,60), font, 1,(255,255,255),2)
        cv2.putText(img,'Similarity ' +str(ret)[:6] ,(10,90), font, 1,(255,255,255),2)

        # Warp back to original view
        unwarp = cv2.warpPerspective(warp_zero, Mi, (warp_zero.shape[1], warp_zero.shape[0]), flags=cv2.INTER_LINEAR)
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, unwarp, 0.3, 0)

        return result
    
    return pipeline

if __name__ == "__main__":
    from calibrate import *
    from transforms import *
    _, _, mtx, dist, _, _ = calibrate_camera(draw_chessboard = True)
    sample_frame = get_sample_clip()
    M, Mi = getPerspectiveTransformMatrices(sample_frame)
    video_pipeline = return_pipeline(mtx, dist, M, Mi)
    # output_video(video_pipeline, input_video="snip.mp4", output_video="snip_out.mp4")
    output_video(video_pipeline, input_video="project_video.mp4", output_video="project_video_out.mp4")
    output_video(video_pipeline, "challenge_video.mp4", "challenge_video_out.mp4")
    output_video(video_pipeline, "harder_challenge_video.mp4", "harder_challenge_video_out.mp4")