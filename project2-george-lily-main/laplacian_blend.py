import sys
import cv2
import numpy as np
import project2_util

def main():

    if len(sys.argv) != 5:
        print('usage: python laplacian_blend.py IMG1 IMG2 MASK RESULT')
        sys.exit(1)
        
    img1 = cv2.imread(sys.argv[1], cv2.IMREAD_ANYCOLOR)
    img2 = cv2.imread(sys.argv[2], cv2.IMREAD_ANYCOLOR)
    mask = cv2.imread(sys.argv[3], cv2.IMREAD_GRAYSCALE)

    result_filename = sys.argv[4]
    
    result = pyr_blend(img1, img2, mask)

    cv2.imwrite(result_filename, result)

    window = 'Laplacian blend result'
    project2_util.setup_window(window)
    project2_util.label_and_wait_for_key(window, result,
                                         'Result (hit a key when done)')

# See Matt Zucker's project notes for algorithm explanation
def pyr_build(img, max_depth=999):
    g = []
    l = []
    # Add initial image
    g.append(img.copy().astype(np.float32))
    
    i = 0
    # Go to max specified depth
    while i < max_depth:
        g_current = g[i]

        # Stop if too small
        if g_current.shape[0] < 16 or g_current.shape[1] < 16:
            break

        # Blur and downsample 
        g_next_downscaled = cv2.pyrDown(g_current)
        height, width, c = g_next_downscaled.shape

        # Upsample downsampled image
        g_next = cv2.pyrUp(g_next_downscaled, dstsize=(width * 2, height * 2))
        # Resize odd-sized images to allow for proper rescaling
        g_current_resized = cv2.resize(g_current, dsize=(width * 2, height * 2))

        diff = g_current_resized - g_next
        
        g.append(g_next_downscaled)
        l.append(diff)

        i = i + 1
    
    l.append(g_next_downscaled)
    
    return l

# See Matt Zucker's project notes for algorithm explanation
def pyr_reconstruct(lp):
    recons = lp[-1].copy()

    for i in range(len(lp) - 2, -1, -1):
        recons_upscaled = cv2.pyrUp(recons)
        height, width, c = lp[i].shape
        recons_upscaled = cv2.resize(recons_upscaled, dsize = (width, height))
        recons = recons_upscaled + lp[i]

    recons_u8 = np.clip(recons, 0, 255).astype(np.uint8)

    return recons_u8

# See Matt Zucker's project notes for algorithm explanation
def pyr_blend(img1, img2, mask):
    # Build pyramids, like the great pyramids of Egypt, or the food pyramid
    lpA = pyr_build(img1, max_depth = 999)
    lpB = pyr_build(img2, max_depth = 999)

    pyr_blended = []

    for i in range(len(lpA)):
        height, width = lpA[i].shape[:2]
        mask_resized = cv2.resize(mask, dsize = (width, height), interpolation = cv2.INTER_AREA)

        # Alpha blend each Laplacian image
        lp = project2_util.alpha_blend(lpA[i], lpB[i], mask_resized)
        pyr_blended.append(lp)

    # Reconstruct blended image
    recons = pyr_reconstruct(pyr_blended)

    return recons

if __name__ == '__main__':
    main()
    
    
