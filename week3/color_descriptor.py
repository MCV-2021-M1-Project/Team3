import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from similarity import compute_similarity as compute_similarity_measure

class ColorDescriptor(object):

    def __init__(self, color_space:str ="lab", metric:str ="lab_3d", scales:int = 3) -> None:
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
        self.metric = metric
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

        hist = cv2.calcHist([channels], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist)
        return hist.flatten()

    def compute_3d_lab_histogram(self, channels:np.ndarray, mask=None):
        """
          Lab definition:
          L*a*b* breaks the colors down based on human perception and the opponent process of vision. 
          The channels are perceptual lightness, and a* and b* which encode red/green and blue/yellow respectively, 

          Then, we will focus on the chromatic representation. Bins has been set empirically
        """

        hist = cv2.calcHist([channels], [0, 1, 2], mask, [2, 24, 24], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist)
        return hist.flatten()

    @staticmethod
    def compute_hist_mask(image, bbox):
        if bbox is None:
            return np.ones(image.shape[:2]).astype(np.uint8)
        else:
            mask = np.ones(image.shape[:2], dtype="uint8")*255
            cv2.rectangle(mask, (bbox[0],bbox[1]), (bbox[2],bbox[3]), 0, -1)

            # display the masked region
            #masked = cv2.bitwise_and(image, image, mask=mask)
            #cv2.imshow("Applying the Mask", masked)
            return mask.astype(np.uint8)

    def compute_3d_tiled_histogram(self, channels: np.ndarray, metric:str, bbox:tuple = None):
        """ input_params
                bbox: tuple containing two tuples, where first tuple is top-left coordinate of the mask
                and second tuple in left-bottom coordinates e.g ((0,0),(250,250)), if bbox is None, it will
                compute the histograms for all the tiles            
        """
        #if the tile size is not defined at init, we will divide the image in 
        histogram = []
        channels = cv2.cvtColor(channels, self.color_space_map[self.color_space])
        mask = self.compute_hist_mask(channels, bbox)
            # substitute the bounding box with the image mean 
            #channels[bbox[1]:bbox[3],bbox[0]:bbox[2]] = channels.mean(axis=(0,1))
        for pyramid_lvl in range(self.scales+1):
            n_blocks = 2 ** pyramid_lvl
            M = channels.shape[0]//n_blocks
            N = channels.shape[1]//n_blocks
            hist_scale = []
            for y in range(0,M * n_blocks ,M):
                for x in range(0, N * n_blocks ,N):
                    #y1 = y + M
                    #x1 = x + N
                    tile = channels[y:y+M,x:x+N]
                    tile_mask = mask[y:y+M,x:x+N]
                    #if there is not bbox param, we compute hist for all the tiles
                    #if bbox is None:
                    hist = self.histogram_function_map[metric](tile, tile_mask)
                    hist_scale.extend(hist)

            histogram.extend(np.stack(hist_scale).flatten())

        return np.stack(histogram).flatten() # Join the histograms and flat them in one dimension array

    def compute_histogram(self, image: np.ndarray, metric: str, plot=False):
        """
        Method to compute the histogram of an image
        """

        image = cv2.cvtColor(image, self.color_space_map[self.color_space])
        
        if not "3d" in metric:
            image = cv2.split(image)

        hist = self.histogram_function_map[metric](image)

        #hist = hist.astype(np.uint8)
        if plot:
            plt.figure()
            plt.title("Histogram")
            plt.xlabel("Bins")
            plt.ylabel("# of Pixels")
            for color_hist in np.split(hist, len(image)):
                plt.plot(color_hist)
            plt.xlim([0, 256])
            plt.show()
        return hist
    
    def compute_image_similarity(self, dataset, similarity_mode, query_img, text_extractor_method):
        result = self.compute_image_multiscale_similarity(dataset, similarity_mode, query_img, text_extractor_method)
        return result

    def compute_simple_image_similarity(self, dataset, similarity_mode, query_img):
        result = []
        query_img_hist = self.compute_histogram(query_img, metric=self.metric)
        for image in dataset.keys():
            image_hist = self.compute_histogram(dataset[image]["image_obj"], metric=self.metric)
            sim_result = compute_similarity_measure(image_hist, query_img_hist,similarity_mode)
            result.append([image, sim_result])
        return result

    def compute_image_multiscale_similarity(self, dataset, similarity_mode, query_img , text_extractor_method: callable):
        result = []
        if text_extractor_method is not None:
            bbox_query ,  _= text_extractor_method(query_img,None,None)
        else:
            bbox_query = None
        query_img_hist = self.compute_3d_tiled_histogram(query_img, self.metric, bbox_query)
        for image in dataset.keys():
            image_hist = self.compute_3d_tiled_histogram(dataset[image]["image_obj"], metric=self.metric)
            sim_result = compute_similarity_measure(image_hist, query_img_hist, similarity_mode)
            result.append([image, sim_result])
        return result
