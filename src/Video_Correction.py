import numpy as np
import cv2

def imhist(im):
    # Finding out the histogram of the given video
    m, n = im.shape
    h = [0.0] * 256
    for i in range(m):
        for j in range(n):
            h[im[i, j]] += 1
    histogram = np.array(h) / (m * n)
    return histogram

def cumsum(a):
    # Calculating the cumulative sum
    return [sum(a[:i + 1]) for i in range(len(a))]


def histeq(im):
    # Calculating the histogram equalization of the given video
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    I = im[:,:,2]
    h = imhist(I)
    cdf = np.array(cumsum(h))
    sk = np.uint8(255 * cdf)
    s1, s2 = I.shape
    img = np.zeros_like(I)
    for i in range(0, s1):
        for j in range(0, s2):
            img[i, j] = sk[I[i, j]]
    im[:,:,2] = img
    im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
    return im

def gamma_correction(image, gamma):
    # Calculating the reverse gamma correction required for the video
    table = np.array([((i / 255) ** gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    m, n, o = image.shape
    gamma_corrected_img = np.zeros_like(image)
    for i in range(m):
        for j in range(n):
            for k in range(o):
                gamma_corrected_img[i, j, k] = table[image[i, j, k]]
    return gamma_corrected_img

cap = cv2.VideoCapture('../media/input/Video.mp4')
ret, frame = cap.read()
D = (frame.shape[1], frame.shape[0])
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
FPS = 25
out = cv2.VideoWriter("../media/output/Output_Video_Correction.avi", fourcc, FPS, D)

while (True):
    ret_val, image = cap.read()
    if not ret_val:
        break
    else:
        # Histogram Equalization
        eq = histeq(image)

        # Reverse Gamma Correction
        gq = gamma_correction(eq, 2.5)

        out.write(gq)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()