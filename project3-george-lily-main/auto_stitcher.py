# Spring 24, ENGR 27: George Fang, Lily Davoren

import sys, os
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join

def main():
    # load all data from command line
    #
    # note: the program currently supports specifying more than two
    # images on the command line but you only need to support 1 max
    # in your project

    if len(sys.argv) != 2 or not os.path.exists(sys.argv[1]):
        print('incorrect arguments!' if len(sys.argv) != 2 else 'path is invalid!')
        print('usage: python auto_stitcher.py PATH')
        print('e.g. python auto_stitcher.py .\\data\\weird\\')
        sys.exit(1)
    
    dataset = []
    image_filenames = [f for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f))]

    for image_filename in image_filenames:
        # Read images from path
        image = cv2.imread(os.path.join(sys.argv[1], image_filename))

        if image is None: 
            print('error loading', image)
            sys.exit(1)

        h, w = image.shape[:2]
        print('loaded {}x{} image from {}'.format(h, w, image_filename))

        # Convert to grayscale
        image_bw = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 

        # Initialize detector algorithm 
        sift = cv2.SIFT_create()

        image_keypoints, image_descriptors = sift.detectAndCompute(image_bw, None)

        dataset.append((image, image_keypoints, image_descriptors))

    # dataset now contains [(imageA, pointsA), (imageB, pointsB)]
    print('ready to stitch images together into data/output.jpg')
    
    # Get images, corners, and points from 2-image dataset
    imgA = dataset[0][0]
    widthA = imgA.shape[0]
    heightA = imgA.shape[1]

    imgB = dataset[1][0]
    widthB = imgB.shape[0]
    heightB = imgB.shape[1]

    cornersA = np.array([[[0, 0]], [[widthA, 0]], [[0, heightA]], [[widthA, heightA]]]).astype(np.float32)
    cornersB = np.array([[[0, 0]], [[widthB, 0]], [[0, heightB]], [[widthB, heightB]]]).astype(np.float32)
    ptsA = dataset[0][1]
    ptsB = dataset[1][1]
    descriptorsA = dataset[0][2]
    descriptorsB = dataset[1][2]

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) # or pass empty dictionary
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    all_matches = flann.knnMatch(descriptorsA,descriptorsB,k=2)
    
    # Store all the good matches per Lowe's ratio test
    # This code adapted from https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
    matches = []
    for m,n in all_matches:
        if m.distance < 0.7*n.distance:
            matches.append(m)

    if len(matches) > 10:
        hptsA = np.float32([ ptsA[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        hptsB = np.float32([ ptsB[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    else:
        print("not enough matches :(")

    # Generate homography using HLS
    # This blog post suggested USAC instead of RANSAC: https://opencv.org/blog/evaluating-opencvs-new-ransacs/
    H, mask = cv2.findHomography(hptsA, hptsB, cv2.USAC_MAGSAC, 5.0)
    # matchesMask = mask.ravel().tolist()
    # Find dimensions of viewport & translation matrix data
    x,y,w,h = cv2.boundingRect(np.concatenate((cornersA, cv2.perspectiveTransform(cornersB, H))))
    # Define translation matrix
    T = np.matrix([[1,0,-x], [0,1,-y], [0,0,1]]).astype(np.float64)
    # Multiply translation with homography for image A
    M = T @ H

    # Transform images using appropriate matrices & retrieved viewport dimenisons
    txformA = cv2.warpPerspective(imgA, M, (w,h))
    txformB = cv2.warpPerspective(imgB, T, (w,h))

    # Create a mask where the images overlap
    ret, maskA = cv2.threshold(cv2.cvtColor(txformA, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    ret, maskB = cv2.threshold(cv2.cvtColor(txformB, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(maskA, maskB)
    not_mask = cv2.bitwise_not(mask)

    # Stitch images using masks
    final_img = np.zeros_like(txformA)
    individual = cv2.bitwise_or(txformA, txformB, mask = not_mask)
    intersect = cv2.bitwise_or(final_img, cv2.addWeighted(txformA, .5, txformB, .5, 1), mask = mask)
    final_img = cv2.add(individual, intersect)

    # # Alpha blend... doesn't really work all the time
    # final_img = alpha_blend(txformA, txformB, mask)

    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #                    singlePointColor = None,
    #                    matchesMask = matchesMask, # draw only inliers
    #                    flags = 2)
    # final_img = cv2.drawMatches(imgA,ptsA,imgB,ptsB,good,None,**draw_params)

    # Save the image
    cv2.imwrite("./data/output.jpg", final_img)
    print("finished!!")

def alpha_blend(img1, img2, mask):
    
    """Perform alpha blend of img1 and img2 using mask.

    Result is an image of same shape as img1 and img2.  Wherever mask
    is 0, result pixel is same as img1. Wherever mask is 255 (or 1.0
    for float mask), result pixel is same as img2. For values in between,
    mask acts as a weight for a weighted average of img1 and img2.

    See https://en.wikipedia.org/wiki/Alpha_compositing
    """

    (h, w) = img1.shape[:2]

    assert img2.shape == img1.shape
    assert mask.shape == img1.shape or mask.shape == (h, w)

    result = np.empty_like(img1)

    if mask.dtype == np.uint8:
        mask = mask.astype(np.float32) / 255.0

    if len(mask.shape) == 2 and len(img1.shape) == 3:
        mask = mask[:, :, None]

    result[:] = img1 * (1 - mask) + img2 * mask

    return result

if __name__ == '__main__':
    main()
    
