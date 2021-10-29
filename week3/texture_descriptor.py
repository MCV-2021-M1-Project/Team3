import cv2
import scipy
import numpy as np
import os
from matplotlib import pyplot as plt
from similarity import compute_similarity
from skimage.feature import local_binary_pattern



class TextureDescriptor(object):

    def __init__(self, color_space: str = "gray", descriptor: str = "HOG", scales: int = 0) -> None:
        self.color_space_map = {
            "gray": cv2.COLOR_BGR2GRAY,
            "rgb": cv2.COLOR_BGR2RGB,
            "hsv": cv2.COLOR_BGR2HSV,
            "lab": cv2.COLOR_BGR2LAB
        }

        self.histogram_function_map = {
            "gray": self.compute_standard_histogram,
            "rgb_1d": self.compute_standard_histogram,
            "rgb_3d": self.compute_3d_rgb_histogram,
            "hsv": self.compute_standard_histogram,
            "lab": self.compute_standard_histogram,
            "lab_3d": self.compute_3d_lab_histogram
        }
        self.descriptor = descriptor
        winSize = (64, 64)
        blockSize = (64, 64)
        blockStride = (64, 64)
        cellSize = (8, 8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        self.descriptor_type = {
            'HOG': cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                     histogramNormType, L2HysThreshold, gammaCorrection, nlevels).compute,
            'DCT': scipy.fft.dctn,
            #'color_histogram': self.compute_3d_rgb_histogram,
            'LBP':self.lbp_histogram,
            'DCT':self.dct_coefficients,            
            # skimage.feature.local_binary_pattern(a,8,1)
        }
        self.color_space = color_space
        self.scales = scales

    def compute_standard_histogram(self, channels: np.ndarray, mask=None):
        hist_chan = []
        for channel in channels:
            hist = cv2.calcHist([channel], [0], mask, [256], [0, 256])
            hist = cv2.normalize(hist, hist)
            hist_chan.append(hist)
        hist = np.concatenate((hist_chan))
        return hist

    def compute_3d_rgb_histogram(self, channels: np.ndarray, mask=None):
        """
            3D color histogram (utilizing all channels) with 8 bins in each direction 
        """

        hist = cv2.calcHist([channels], [0, 1, 2], mask, [
                            8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist)
        return hist.flatten()

    def compute_3d_lab_histogram(self, channels: np.ndarray, mask=None):
        """
          Lab definition:
          L*a*b* breaks the colors down based on human perception and the opponent process of vision. 
          The channels are perceptual lightness, and a* and b* which encode red/green and blue/yellow respectively, 

          Then, we will focus on the chromatic representation. Bins has been set empirically
        """

        hist = ([channels], [0, 1, 2], mask, [
                2, 24, 24], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist)
        return hist.flatten()

    def compute_hist_mask(image, bbox):
        if bbox is None:
            return np.ones(image.shape[:2]).astype(np.uint8)
        else:
            mask = np.ones(image.shape[:2], dtype="uint8")*255
            cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 0, -1)

            # display the masked region
            #masked = cv2.bitwise_and(image, image, mask=mask)
            #cv2.imshow("Applying the Mask", masked)
            return mask.astype(np.uint8)

    def compute_descriptor(self, channels: np.ndarray, bbox: tuple = None):
        """ input_params
                bbox: tuple containing two tuples, where first tuple is top-left coordinate of the mask
                and second tuple in left-bottom coordinates e.g ((0,0),(250,250)), if bbox is None, it will
                compute the histograms for all the tiles            
        """
        # if the tile size is not defined at init, we will divide the image in
        histogram = []
        channels = cv2.cvtColor(
            channels, self.color_space_map[self.color_space])
        if bbox is not None:
            mask = self.compute_hist_mask(channels, bbox)
            # substitute the bounding box with the image mean
            #channels[bbox[1]:bbox[3],bbox[0]:bbox[2]] = channels.mean(axis=(0,1))
        for pyramid_lvl in range(self.scales+1):
            n_blocks = 2 ** pyramid_lvl
            M = channels.shape[0]//n_blocks
            N = channels.shape[1]//n_blocks
            hist_scale = []
            for y in range(0, M * n_blocks, M):
                for x in range(0, N * n_blocks, N):
                    tile = channels[y:y+M, x:x+N]
                    if bbox is not None:
                        tile_mask = mask[y:y+M, x:x+N]
                    # if there is not bbox param, we compute hist for all the tiles
                    # if bbox is None:
                    if bbox is not None:
                        hist = self.descriptor_type[self.descriptor](tile, mask)
                    else:
                        hist = self.descriptor_type[self.descriptor](tile)
                    hist_scale.extend(hist)

            histogram.extend(np.stack(hist_scale).flatten())

        # Join the histograms and flat them in one dimension array
        return np.stack(histogram).flatten()
    def lbp_histogram(self,image:np.ndarray, points:int=24, radius:float=3.0, bins:int=8, mask:np.ndarray=None) -> np.ndarray:
        # image --> grayscale --> lbp --> histogram
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = (local_binary_pattern(image, points, radius, method="uniform")).astype(np.uint8)

        bins = points + 2
        hist = cv2.calcHist([image],[0], mask, [bins], [0, bins])
        hist = cv2.normalize(hist, hist)
        return hist.flatten()
    def dct_coefficients(self,image:np.ndarray, bins:int=8, mask:np.ndarray=None, num_coeff:int=4) -> np.ndarray:

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if mask is not None:
            image = cv2.bitwise_and(image, image, mask=mask)
            
        block_dct = cv2.dct(np.float32(image)/255.0)

        def _compute_zig_zag(a):
            return np.concatenate([np.diagonal(a[::-1,:], k)[::(2*(k % 2)-1)] for k in range(1-a.shape[0], a.shape[0])])
        
        features = _compute_zig_zag(block_dct[:6,:6])[:num_coeff]
        return features
    def compute_image_similarity(self, image_dataset, similarity_mode, query_img, text_extractor_method: callable):
        result = []
        if text_extractor_method is not None:
            bbox_query,  _ = text_extractor_method(query_img, None, None)
            if bbox_query and self.descriptor != 'color_histogram':
                print('Only can use bbox with color histogram!!')
                exit()
            query_img_hist = self.compute_descriptor(query_img, bbox_query)
        else:
            query_img_hist = self.compute_descriptor(query_img)
        for image in image_dataset.keys():
            image_hist = self.compute_descriptor(image_dataset[image]["image_obj"])
            min_size = min(image_hist.shape[0], query_img_hist.shape[0])
            image_hist = image_hist[:min_size]
            query_img_hist = query_img_hist[:min_size]
            sim_result = compute_similarity(
                image_hist, query_img_hist, similarity_mode)
            result.append([image, sim_result])
        return result


if __name__ == "__main__":
    img = cv2.imread(
        '/home/marcelo/Documents/Master_CV/M1/Team3/datasets/BBDD/bbdd_00000.jpg')
    print(img)
    query_image = cv2.imread(
        '/home/marcelo/Documents/Master_CV/M1/Team3/datasets/BBDD/bbdd_00000.jpg')
    descriptor = TextureDescriptor()
    image_dataset = {}
    image_dataset[1] = {
        "image_name": "1",
        "image_obj": img
    }
    descriptor.compute_image_similarity(
        image_dataset, 'cosine_similairty', img, None
    )
