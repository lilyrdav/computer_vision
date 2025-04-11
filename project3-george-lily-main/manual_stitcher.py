# Spring 24, ENGR 27: George Fang, Lily Davoren

import sys, os
import numpy as np
import cv2

def main():

    # load all data from command line
    #
    # note: the program currently supports specifying more than two
    # images on the command line but you only need to support 1 max
    # in your project

    if len(sys.argv) < 3:
        print('usage: python stitcher.py IMAGE1 IMAGE2')
        sys.exit(1)

    dataset = []
    num_points = None

    for image_filename in sys.argv[1:]:

        image = cv2.imread(image_filename)

        if image is None: 
            print('error loading', image)
            sys.exit(1)

        h, w = image.shape[:2]
        print('loaded {}x{} image from {}'.format(h, w, image_filename))

        basename, _ = os.path.splitext(image_filename)

        points_filename = basename + '.txt'

        try:
            points = np.genfromtxt(points_filename)
        except OSError:
            print('error reading', points_filename)
            sys.exit(1)

        assert len(points.shape) == 2 and points.shape[1] == 2

        if num_points is None:
            num_points = len(points)
        else:
            assert num_points == len(points)

        # turn into shape (n, 1, 2) that findHomography expects
        points = points.reshape(num_points, 1, 2)

        # make it float32
        points = points.astype(np.float32)

        print('read {} points from {}'.format(len(points), points_filename))
        print()

        dataset.append((image, points))


    # dataset now contains [(imageA, pointsA), (imageB, pointsB)]
    print('ready to stitch images together into output.jpg')

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

    # Generate homography using HLS
    H, mask = cv2.findHomography(ptsA, ptsB)
    # Find dimensions of viewport & translation matrix data
    x,y,w,h = cv2.boundingRect(np.concatenate((cornersA, cv2.perspectiveTransform(cornersB, H))))
    # Define translation matrix
    T = np.matrix([[1,0,-x], [0,1,-y], [0,0,1]]).astype(np.float64)
    # Multiply translation with homography for image A
    M = T @ H

    # Transform images using appropriate matrices & retrieved viewport dimenisons
    txformA = cv2.warpPerspective(imgA, M, (w,h))
    txformB = cv2.warpPerspective(imgB, T, (w,h))

    # Average the images as described in instructions
    finalImage = txformA // 2 + txformB // 2

    # Save the image
    cv2.imwrite("output.jpg", finalImage)

if __name__ == '__main__':
    main()
    
