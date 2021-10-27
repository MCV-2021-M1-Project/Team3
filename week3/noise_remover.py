import cv2
import numpy as np 
import math
import numpy as np
from scipy.signal import convolve2d


class NoiseRemover(object):


    def __init__(self, noise_thresh=10) -> None:
        self.noise_thresh = noise_thresh 

    def estimate_noise(self, img):
        """
        Reference: J. Immerkær, “Fast Noise Variance Estimation”, 
        Computer Vision and Image Understanding, Vol. 64, No. 2, 
        pp. 300-302, Sep. 1996

        Args:
            image ([type]): [description]

        Returns:
            [type]: [description]
        """

        H, W = img.shape

        M = [[1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1]]

        sigma = np.sum(np.sum(np.absolute(convolve2d(img, M))))
        sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

        return sigma

    def is_noisy_img(self, img):
        sigma = self.estimate_noise(img)
        return sigma >= self.noise_thresh
    
    def median_filter(self, img, kernel):
      result_img = cv2.bilateralFilter(img, kernel,51,51)
      return result_img

    def remove_noise(self, img, filter_type, kernel_size):
      noise_filter = {
        "bilateral": self.median_filter,
        "median": cv2.medianBlur,
      }
      result_img = noise_filter[filter_type](img, kernel_size)
      return result_img
    

if __name__ == "__main__":
    noiser_remover = NoiseRemover(noise_thresh=10)
    img = cv2.imread("datasets/qsd1_w3/00001.jpg") 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    is_noisy = noiser_remover.is_noisy_img(img_gray)
    print(is_noisy)
    if is_noisy:
        denoised_img = noiser_remover.remove_noise(img, "median", 3)
    cv2.imwrite("datasets/qsd1_w3/00001_filter.jpg", denoised_img)
    #image = cv2.imread(args["image"])
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)