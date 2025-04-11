########################################################################
#
# File:   blob_track.py
# Author: George Fang, Lily Davoren
# Date:   February 2024
#
# Written for ENGR 27 - Computer Vision
#
########################################################################

import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

# Constants for object tracking
MIN_OBJECT_AREA = 50 # Minimum px area to be considered an object
MAX_OBJECT_JUMP_DISTANCE = 50 # Max distance an object can move between frames to be considered the same object
OBJECT_PERMANENCE_THRESHOLD = 3 # Maximum frames object can be missing before it considered "gone"

# Class to keep track of each object's information
class ObjectData:
    count = 1 # How many objects have been created
    def __init__(self):
        self.id = ObjectData.count # Unique id for each object
        ObjectData.count = ObjectData.count + 1

        self.color = None
        self.locations = []
        self.missing = 0
        self.gone = False

objs_being_tracked : list[ObjectData] = [] # List to store objects being tracked

# Actually run the code
def main():
    # Check input file
    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
    else:
        print('Specify a video name...')
        sys.exit(1)

    # Start video capture
    capture = cv2.VideoCapture(input_filename)
    ok, img_current_frame = capture.read()

    # Read in file; bail if error.
    if capture:
        print('Opened file')
    else:
        print('Error opening video capture!')
        sys.exit(1)

    # Fetch the first frame; bail if none.
    ok, frame = capture.read()

    if not ok or frame is None:
        print('No frames in video')
        sys.exit(1)
    
    width = img_current_frame.shape[1]
    height = img_current_frame.shape[0]
    wait = 20

    # Didn't end up being super helpful because I didn't have enough time to fully make it work
    # img_temporal_avg = calc_temporal_average(cv2.VideoCapture(input_filename))
    # # Blur the averaged image to reduce bias for objects that spent a lot of time in place
    # img_temporal_avg_blur = cv2.GaussianBlur(img_temporal_avg, (0,0), 10)
    # cv2.imshow('Temporal Averaged Image', img_temporal_avg)
    # cv2.imshow('Gaussian Blurred Avg Image', img_temporal_avg_blur)

    global target_locations # List of (x,y) locations to find target colors
    target_locations = set() 
    target_colors = set()

    img_select_colors = img_current_frame.copy() # Copy of the current frame to select colors from

    print("\n\nLEFT CLICK the brightest part of the object(s) you want to track")
    print("Press C to clear your selection")
    print("Press SPACEBAR when finished")
    print("Press ESC to end the simulation at any time!")
    cv2.namedWindow('Select Object(s) - see console') # Create a window
    cv2.setMouseCallback('Select Object(s) - see console', get_mouse_location) # Set the callback function
    
    # Allow user to select colors to track
    while(True):
        cv2.imshow('Select Object(s) - see console', img_select_colors) # Display the image

        for (x,y) in target_locations:
            target_colors.add(tuple(int(item) for item in img_current_frame[y, x])) # Add the color to the list
            print("Recorded mouse click at {:d}, {:d} so {} is a target color".format(x,y, img_current_frame[y, x]))
        
        target_locations.clear()

        # Actually do the color filtering process to show the user the color they're selecting
        img_colors_threshold = None 

        for color in target_colors:
            # Filter out item by color
            mask_one_color_threshold = np.linalg.norm(img_current_frame.astype(np.float32) - color, axis=2) < 100
            img_one_color_threshold = mask_one_color_threshold.astype(np.uint8) * 255 # Convert to img

            img_colors_threshold = img_colors_threshold | img_one_color_threshold if img_colors_threshold is not None else img_one_color_threshold
        
        if target_colors:
            img_threshold_morph = cv2.morphologyEx(src=img_colors_threshold, op=cv2.MORPH_OPEN, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
            contours, hierarchy = cv2.findContours(img_threshold_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_select_colors, contours, -1, (0,255,0), thickness=3)
        else:
            img_select_colors = img_current_frame.copy()

        key = cv2.waitKey(200) & 0xFF # Wait for a key press
        if key == 27 or key == ord(' '):
            break
        elif key == ord('c'):
            target_colors.clear()

    if target_colors:
        print("\n\nPress A to slow down the video")
        print("Press S to go frame by frame")
        print("Press D to speed up the video")
        print("Press SPACEBAR to play/pause the video")
    else:
        print('You did not select any colors to track! Bye bye...') # Bail if no colors were selected
        sys.exit(1)

    # Main loop for processing the video frames
    while ok and img_current_frame is not None:
        img_colors_threshold = None

        for color in target_colors:
            # Filter out item by color
            mask_one_color_threshold = np.linalg.norm(img_current_frame.astype(np.float32) - color, axis=2) < 100
            img_one_color_threshold = mask_one_color_threshold.astype(np.uint8) * 255 # Convert to img

            img_colors_threshold = img_colors_threshold | img_one_color_threshold if img_colors_threshold is not None else img_one_color_threshold

        # Didn't end up being super helpful because I didn't have enough time to fully make it work
        # img_rgb_diff = cv2.absdiff(img_current_frame, img_temporal_avg).max(axis=2)
        # mask_rgb_threshold = img_rgb_diff > 160
        # img_rgb_threshold = mask_rgb_threshold.astype(np.uint8) * 255 # Convert to img

        # Use open morphological operator to remove noise
        img_threshold_morph = cv2.morphologyEx(src=img_colors_threshold, op=cv2.MORPH_OPEN, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))

        # Find the contours
        contours, hierarchy = cv2.findContours(img_threshold_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw countours on image
        img_contours = img_current_frame.copy()
        cv2.drawContours(img_contours, contours, -1, (0,255,0), thickness=3)
        track_objects(img_contours, contours)

        # Display live video feed with filters
        cv2.imshow("Contour Tracking", img_contours)
        cv2.imshow('Color Thresholded', img_colors_threshold.astype(np.uint8))
        cv2.imshow('Thresholded + Morphed', img_threshold_morph)
        # cv2.imshow('Original', img_current_frame)
        # cv2.imshow('Color Removed', img_color_diff.astype(np.uint8))
        # cv2.imshow('Background Removed', img_rgb_diff.astype(np.uint8))
        # cv2.imshow('RGB Thresholded', img_rgb_threshold)

        # Check for space hit and wait for some ms:
        key = cv2.waitKey(wait) & 0xFF
        if key == 27: # Escape button
            break # Exit program
        elif key == ord(' '): # Pause button
            cv2.waitKey(-1)
            print("X (Paused) Press SPACEBAR to continue")
        elif key == ord('a'): # Slow down
            wait += 10
            print("- (Slowed down) Added 10 to delay")
        elif key == ord('d'): # Speed up
            if wait >= 20:
                wait -= 10
                print("+ (Sped up) Subtracted 10 from delay")
        elif key == ord('s'): # Go frame by frame
            wait = -10
            print("~ (Going frame by frame) Press S to step each frame")

        # Get next frame
        ok, img_current_frame = capture.read(img_current_frame)

    print("In total, {:d} objects were tracked".format(len(objs_being_tracked)))
    plot_data(width,height)


# -----------------------------------------------------------------
# Helper methods
# -----------------------------------------------------------------

# Track contours and update tracked objects 
def track_objects(display, contours):
    found_objects = []

    for contour in contours:
        info = get_contour_info(contour)
        location = info['mean']
        area = info['area']

        # If the area is large enough, track the object
        if area > MIN_OBJECT_AREA:
            obj = calc_closest_object(objs_being_tracked, location, maxDistance=MAX_OBJECT_JUMP_DISTANCE) if objs_being_tracked else None

            # If the object is not being tracked, create a new object
            if obj is None:
                obj = ObjectData()
                objs_being_tracked.append(obj)
                obj.color = tuple(int(x) for x in display[int(location[1]), int(location[0])]) # Weird requirement for the color to be a fresh tuple
                print("Tracking new object with ID", obj.id)

            found_objects.append(obj) # Mark as found
            obj.locations.append(location) # Add current location to running list
    
    missing_objects = list(set(objs_being_tracked) - set(found_objects)) # List objects not found in current frame

    # Increment the missing count for the missing objects
    for missing_obj in missing_objects:
        missing_obj.missing = missing_obj.missing + 1

    # Remove the missing objects from the list of objects being tracked
    for obj in objs_being_tracked:
        if not obj.gone and obj.missing >= OBJECT_PERMANENCE_THRESHOLD:
            obj.gone = True
            print("Uh oh, object {:d} is gone".format(obj.id))
        for location in obj.locations:
            cv2.circle(display, ([int(location[0]), int(location[1])]), radius=1, color=obj.color, thickness=1, lineType=cv2.LINE_AA)

        draw_outlined_text(display, 'Object {:d}'.format(obj.id), obj.locations[-1] + (-5 -10))
            

# Find the closest object to the current location
def calc_closest_object(objects : list[ObjectData], currentLocation, maxDistance):
    candidate = None

    for obj in objects:
        # If the current array item better than candidate
        if not obj.gone \
            and (candidate is None or np.linalg.norm(currentLocation - obj.locations[-1]) < np.linalg.norm(currentLocation - candidate.locations[-1])):
            
            candidate = obj
            # candidate_change_in_loc = obj_est_loc
    
    # Ignore if candidate is too far from the object location
    return candidate if np.linalg.norm(currentLocation - candidate.locations[-1]) <= maxDistance else None

# Plot it all it matplotlib as well
def plot_data(width, height):
    for obj in objs_being_tracked:
        y = [location[1] for location in obj.locations]
        x = [location[0] for location in obj.locations]
        plt.plot(x, y, label='Object {:d}'.format(obj.id))
    
    axis = plt.gca()
    axis.set_xlim([0, width])
    axis.set_ylim([0, height])
    axis.invert_yaxis()
    
    plt.show()

def calc_temporal_average(capture):
    # Read in file; bail if error.
    if capture:
        print('Opened file')
    else:
        print('Error opening video capture!')
        sys.exit(1)

    # Fetch the first frame; bail if none.
    ok, frame = capture.read()

    if not ok or frame is None:
        print('No frames in video')
        sys.exit(1)
    
    count = 0
    accum = np.zeros_like(frame, dtype=np.float32)

    while ok and frame is not None:
        accum += frame
        count += 1
        ok, frame = capture.read(frame)

    return np.clip(accum/count, 0, 255).astype(np.uint8)

# Event listener to allow people to click on screen
def get_mouse_location(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        target_locations.add((x,y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        untarget_locations.add((x,y))

# -----------------------------------------------------------------
# Helper methods borrowed from Matt Zucker's regions.py
# -----------------------------------------------------------------
        
def make_point(arr):
    return tuple(np.round(arr).astype(int).flatten())

def draw_outlined_text(img, text, location):
    # width/color pairs for drawing white over black outlines
    DRAW_OUTLINED = [ (3, (0, 0, 0)), (1, (255, 255, 255)) ]
    
    for width, color in DRAW_OUTLINED:
        cv2.putText( img, text, make_point(location), 
                     cv2.FONT_HERSHEY_PLAIN,
                     0.8, color, width, cv2.LINE_AA )

######################################################################
#
# Compute moments and derived quantities such as mean, area, and
# basis vectors from a contour as returned by cv2.findContours.
#
# Feel free to use this function with attribution in your project 1
# code.
#
# Returns a dictionary.

def get_contour_info(c):

    # For more info, see
    #  - https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
    #  - https://en.wikipedia.org/wiki/Image_moment

    m = cv2.moments(c)

    s00 = m['m00']
    s10 = m['m10']
    s01 = m['m01']
    c20 = m['mu20']
    c11 = m['mu11']
    c02 = m['mu02']

    if s00 != 0:

        mx = s10 / s00
        my = s01 / s00

        A = np.array( [
                [ c20 / s00 , c11 / s00 ],
                [ c11 / s00 , c02 / s00 ] 
                ] )

        W, U, Vt = cv2.SVDecomp(A)

        ul = 2 * np.sqrt(W[0,0])
        vl = 2 * np.sqrt(W[1,0])

        ux = ul * U[0, 0]
        uy = ul * U[1, 0]

        vx = vl * U[0, 1]
        vy = vl * U[1, 1]

        mean = np.array([mx, my])
        uvec = np.array([ux, uy])
        vvec = np.array([vx, vy])

    else:
        
        mean = c[0].astype('float')
        uvec = np.array([1.0, 0.0])
        vvec = np.array([0.0, 1.0])

    return {'moments': m, 
            'area': s00, 
            'mean': mean,
            'b1': uvec,
            'b2': vvec}

##################################################################
# Unused code that was interesting but not helpful
# # Get the most common color in image
# def find_common_color(a):
#     colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True) # Count the number of unique colors
#     return colors[count.argmax()] # Return the most common color

if __name__ == '__main__':
    main()
