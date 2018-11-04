import cv2
import numpy as np
import matplotlib.pyplot as plt
def do_pca(x, topk):
    mean = np.mean(x, axis=0)[np.newaxis, ...]
    # std = np.std(x, axis=0)[np.newaxis, ...]
    # x = (x - mean)/std
    x = x - mean
    xtx = np.dot(x.transpose(), x)
    e, p = np.linalg.eig(xtx)
    indx = np.argsort(e)[::-1]
    lp = p[:, indx[:topk]]
    lx = np.dot(x, lp)
    # lx = np.dot(lx, lp.transpose())*std + mean
    lx = np.dot(lx, lp.transpose()) + mean
    return lx.astype(np.uint8)

def do_hist(x):
    hist, bins = np.histogram(x.ravel(), bins=255)
    cdf = hist.cumsum()
    cdf = 255 * cdf / cdf[-1]
    im = np.interp(x.ravel(), bins[:-1], cdf)
    return im.astype(np.uint8).reshape(x.shape), cdf

def test_pca():
    im1 = cv2.imread('imgs/l_hires.jpg', cv2.IMREAD_GRAYSCALE)
    im2 = do_pca(im1, int(im1.shape[1]/5))
    plt.imshow(im2, cmap='gray')
    plt.show()

if __name__ == '__main__':
    im1 = cv2.imread('imgs/l_hires.jpg', cv2.IMREAD_GRAYSCALE)
    im2 = do_hist(im1)
    plt.imshow(im2, cmap='gray')
    plt.show()