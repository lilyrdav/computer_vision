import sys
import cv2
import numpy as np
import hybrid
import project2_util
import laplacian_blend

def main():
    if len(sys.argv) != 5:
        print('usage: python face_detect_hybrid.py IMG1 IMG2 SIGMA K')
        sys.exit(1)

    # Use facial recognition model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Read input parameters
    img1 = cv2.imread(sys.argv[1])
    img2 = cv2.imread(sys.argv[2])
    sigma = float(sys.argv[3])
    k = float(sys.argv[4])

    # Convert into grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Detect faces
    img1_faces = face_cascade.detectMultiScale(img1_gray, 1.1, 4, minSize=(100,100))
    img2_faces = face_cascade.detectMultiScale(img2_gray, 1.1, 4, minSize=(100,100))
    # Draw face mask using some trial-and-error constants
    # This will not work for multiple faces
    for (x, y, w, h) in img1_faces:
        continue
    for (x2, y2, w2, h2) in img2_faces:
        # cv2.ellipse(img2, (x2 + w2//2, y2 + h2//2), (w2//5 + w2//5, h2//4 + h2//5), 0, 0, 360, (255, 255, 255), -1)

        # Failed attempt to implement scaled implementation
        # scale = (w*h)/(w2*h2)
        # midpoint_x = int(scale * (x2 + w2//2))
        # midpoint_y = int(scale * (y2 + h2//2))
        # img2_scaled = cv2.resize(img2, dsize = (int(scale * img2.shape[1]), int(scale * img2.shape[0])), interpolation = cv2.INTER_AREA)
        # img2_cropped = img2_scaled[0:img1.shape[0], 0:img1.shape[1]]
        img2_shifted = translate_image(img2, x+w//2-(x2+w2//2), y+h//2-(y2+h2//2))

    hybrid_image = hybrid.hybrid_image(img1.astype(np.float32), img2_shifted.astype(np.float32), sigma, k)

    pyr = laplacian_blend.pyr_build(hybrid_image)
    cv2.imwrite("tests.jpg", project2_util.visualize_pyramid(pyr))

    
    # Display the output
    window = 'Hybrid blend image result'
    project2_util.setup_window(window)
    project2_util.label_and_wait_for_key(window, img1,
                                         'Image 1 (hit a key when done)')
    project2_util.label_and_wait_for_key(window, img2,
                                         'Image 2 (hit a key when done)')
    project2_util.label_and_wait_for_key(window, hybrid_image,
                                         'Result (hit a key when done)')

def translate_image(image, x, y):
    # Store height and width of the image 
    height, width = image.shape[:2] 
    
    T = np.float32([[1, 0, x], [0, 1, y]]) 
    
    # We use warpAffine to transform 
    # the image using the matrix, T 
    return cv2.warpAffine(image, T, (width, height)) 

if __name__ == '__main__':
    main()