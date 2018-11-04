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

if __name__ == '__main__':
    img = cv2.imread('imgs/l_hires.jpg', cv2.IMREAD_GRAYSCALE)
    img_pca = do_pca(img, int(img.shape[1]/5))
    plt.imshow(img_pca, cmap='gray')
    plt.show()