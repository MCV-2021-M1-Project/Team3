import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


class ColorDescriptor(object):

    def __init__(self, , color_space:str ="gray", scales:int = 3) -> None:
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
        self.color_space = color_space
        self.scales = scales


    def compute_standard_histogram(self, channels: np.ndarray):
        hist_chan = []
        for channel in channels:
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist)
            hist_chan.append(hist)
        hist = np.concatenate((hist_chan))
        return hist

    def compute_3d_rgb_histogram(self, channels: np.ndarray):
        """
            3D color histogram (utilizing all channels) with 8 bins in each direction 
        """

        hist = cv2.calcHist([channels], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist)
        return hist.flatten()

    def compute_3d_tiled_histogram(self, channels: np.ndarray, metric:str, bbox:tuple = None):
        """ input_params
                bbox: tuple containing two tuples, where first tuple is top-left coordinate of the mask
                and second tuple in left-bottom coordinates e.g ((0,0),(250,250)), if bbox is None, it will
                compute the histograms for all the tiles            
        """
        #if the tile size is not defined at init, we will divide the image in 
        histogram = []
        channels = cv2.cvtColor(channels, self.color_space_map[self.color_space])

        if bbox is not None:
            # substitute the bounding box with the image mean 
            #cv2.imwrite("before.png", channels)
            channels[bbox[1]:bbox[3],bbox[0]:bbox[2]] = channels.mean(axis=(0,1))
            #cv2.imwrite("after_fliped.png", channels)
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
                    #if there is not bbox param, we compute hist for all the tiles
                    #if bbox is None:
                    hist = self.histogram_function_map[metric](tile)
                    hist_scale.extend(hist)

            histogram.extend(np.stack(hist_scale).flatten())

        return np.stack(histogram).flatten() # Join the histograms and flat them in one dimension array

    def compute_3d_lab_histogram(self, channels:np.ndarray):
        """
          Lab definition:
          L*a*b* breaks the colors down based on human perception and the opponent process of vision. 
          The channels are perceptual lightness, and a* and b* which encode red/green and blue/yellow respectively, 

          Then, we will focus on the chromatic representation. Bins has been set empirically
        """

        hist = cv2.calcHist([channels], [0, 1, 2], None, [2, 24, 24], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist)
        return hist.flatten()

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

    def compute_image_similarity(self, query_img, metric:str ):
        result = []
        query_img_hist = self.compute_histogram(query_img, metric=metric)
        for image in self.image_dataset.keys():
            image_hist = self.compute_histogram(self.image_dataset[image]["image_obj"], metric=metric)
            sim_result = compute_similarity(image_hist, query_img_hist,self.similarity_mode)
            result.append([image, sim_result])
        return result

    def compute_image_multiscale_similarity(self, query_img, metric:str , text_extractor_method: callable):
        result = []
        bbox_query ,  _= text_extractor_method(query_img,None,None)
        query_img_hist = self.compute_3d_tiled_histogram(query_img, metric, bbox_query)
        for image in self.image_dataset.keys():
            image_hist = self.compute_3d_tiled_histogram(self.image_dataset[image]["image_obj"], metric)
            sim_result = compute_similarity(image_hist, query_img_hist,self.similarity_mode)
            result.append([image, sim_result])
        return result
