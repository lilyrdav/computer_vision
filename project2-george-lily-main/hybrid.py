import sys
import cv2
import numpy as np
import project2_util

def main():
    if len(sys.argv) != 6:
        print('usage: python hybrid.py IMG1 IMG2 SIGMA K RESULT')
        sys.exit(1)

    img1 = cv2.imread(sys.argv[1], cv2.IMREAD_ANYCOLOR).astype(np.float32)
    img2 = cv2.imread(sys.argv[2], cv2.IMREAD_ANYCOLOR).astype(np.float32)

    sigma = float(sys.argv[3])
    k = float(sys.argv[4])

    result = hybrid_image(img1,img2, sigma, k)

    result_filename = sys.argv[5]

    cv2.imwrite(result_filename, result)

    window = 'Hybrid image result'
    project2_util.setup_window(window)
    project2_util.label_and_wait_for_key(window, result,
                                         'Result (hit a key when done)')
    

def hybrid_image(img1, img2, sigma, k):
    A_lopass = cv2.GaussianBlur(img1, ksize=(0,0), sigmaX=sigma)
    B_hipass = img2 - cv2.GaussianBlur(img2, ksize=(0,0), sigmaX=sigma * k)
    return np.clip(A_lopass + B_hipass, 0, 255).astype(np.uint8)


if __name__ == '__main__':
    main()
    
    
