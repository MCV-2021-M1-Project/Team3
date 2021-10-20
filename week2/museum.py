import os
from PIL import Image, ImageChops
import cv2
from matplotlib import pyplot as plt
import numpy as np
import shapely.geometry as geom
from similarity import compute_similarity

VALID_IMAGE_FORMATS = ['JPEG']


class FileIsNotImageError(Exception):
    """The provided file does not match an Image type"""
    pass

class Museum(object):
    """
    A class for loading the museum images and 
    extrat its feature descriptors.
    """

    def __init__(self, data_set_path: str, similarity_mode: str ="L1_norm", color_space:str ="gray", rm_frame:bool = False, scales:int = 3):
        self.rm_frame = rm_frame
        self.image_dataset = self.load_images_dataset(data_set_path)
        self.similarity_mode = similarity_mode
        self.color_space = color_space
        self.scales = scales

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

    def load_images_dataset(self, image_path: str, ):
        """
        Method to load the image from dataset path
        """
        image_dataset = {}
        for image_name in os.listdir(image_path):
            if self.file_is_image(os.path.join(image_path, image_name)):
                image = cv2.imread(os.path.join(image_path, image_name))
                if self.rm_frame:
                    image = self.remove_frame(image)
                image_num = self.get_image_number(image_name)
                image_dataset[image_num] = {
                    "image_name": image_name,
                    "image_obj": image
                }
        return image_dataset
    
    def load_query_img(self, image_path: str):
        if self.file_is_image(image_path):
            image = cv2.imread(image_path)
            if self.rm_frame:
                image = self.remove_frame(image)
            return image
        else:
            raise FileIsNotImageError("The provided file does not match an Image type")
    
    def compute_similarity(self, image_set: str, metric:str):
        set_result = []
        if os.path.isdir(image_set):
            for image in os.listdir(image_set):
                try:
                    set_result.append(self.compute_image_similarity(os.path.join(image_set,image), metric=metric))
                except FileIsNotImageError:
                    pass
        else:
            set_result = self.compute_image_similarity(image_set, metric=metric)
        return set_result

    def compute_image_similarity(self, image_path: str, metric:str ):
        result = []
        query_img = self.load_query_img(image_path)  
        query_img_hist = self.compute_histogram(query_img, metric=metric)
        for image in self.image_dataset.keys():
            image_hist = self.compute_histogram(self.image_dataset[image]["image_obj"], metric=metric)
            sim_result = compute_similarity(image_hist, query_img_hist,self.similarity_mode)
            result.append([image, sim_result])
        return result

    def compute_image_multiscale_similarity(self, image_path: str, metric:str ):
        result = []
        query_img = self.load_query_img(image_path)
        print(image_path)
        query_img_hist = self.compute_3d_tiled_histogram_given_mask(query_img, metric=metric)
        for image in self.image_dataset.keys():
            image_hist = self.compute_3d_tiled_histogram_given_mask(self.image_dataset[image]["image_obj"], metric=metric)
            sim_result = compute_similarity(image_hist, query_img_hist,self.similarity_mode)
            result.append([image, sim_result])
        return result

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

    def compute_3d_tiled_histogram_given_mask(self, channels: np.ndarray, metric:str, bbox:tuple = None):
        """ input_params
                bbox: tuple containing two tuples, where first tuple is top-left coordinate of the mask
                and second tuple in left-bottom coordinates e.g ((0,0),(250,250)), if bbox is None, it will
                compute the histograms for all the tiles            
        """
        #if the tile size is not defined at init, we will divide the image in 
        histogram = []
        channels = cv2.cvtColor(channels, self.color_space_map[self.color_space])

        for pyramid_lvl in range(self.scales+1):
            n_blocks = 2 ** pyramid_lvl
            M = channels.shape[0]//n_blocks
            N = channels.shape[1]//n_blocks
            hist_scale = []
            for y in range(0,M * n_blocks ,M):
                for x in range(0, N * n_blocks ,N):
                    y1 = y + M
                    x1 = x + N
                    tile = channels[y:y+M,x:x+N]
                    #if there is not bbox param, we compute hist for all the tiles
                    if bbox is None:
                        hist = self.histogram_function_map[metric](tile)
                        hist_scale.extend(hist)
                    else:
                        #create shapely polygon to compute intersection
                        p1 = geom.Polygon([(x,y), (x1,y),(x1,y1),(x,y1),(x,y)])
                        p2 = geom.Polygon([(bbox[0][0],bbox[0][1]), (bbox[1][0],bbox[0][1]),(bbox[1][0],bbox[1][1]),(bbox[0][0],bbox[1][1]),(bbox[0][0],bbox[0][1])])
                        # if it intersects, we must avoid that tile in the histogram, so we append a vector of 0 equals to histogram size, this is needed bcs if not,
                        # we will not be able to compare with label image, can not compute similarity of vectors with different sizes.
                        #ATENTION! intersection is not the same as containing
                        if p1.intersects(p2):
                            histogram = np.concatenate((histogram,np.zeros(8*8*8)), axis=0)
                        else:
                            hist = self.histogram_function_map[metric](tile)
                            histogram = np.concatenate((histogram,hist))
            histogram.extend(np.stack(hist_scale).flatten())

        return np.stack(histogram).flatten()

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

    @staticmethod
    def get_image_number(image_name:str):
        """
        Method to get the image number
        image_name(str)
        """
        image_num = image_name.split("_")[1].split(".")[0]
        return int(image_num)

    @staticmethod
    def file_is_image(image_path:str):
        """
        Method to check that a file is valid image 
        """
        try:
            img = Image.open(image_path)
            if img.format in VALID_IMAGE_FORMATS:
                return True
            else:
                return False
        except Exception:
            return False
    
    def get_image_dataset(self):
        return self.image_dataset

    @staticmethod
    def remove_frame(image: np.ndarray, mark_perc:int =5):
        """
        Method to remove the marc given a fixed number of pixels
        """
        pixels_to_remove = [
            image.shape[0] * mark_perc // 100, 
            image.shape[1] * mark_perc // 100
        ]
        image = image[
            pixels_to_remove[0]:image.shape[0]-pixels_to_remove[0], 
            pixels_to_remove[1]:image.shape[1]-pixels_to_remove[1]
        ]
        return image


if __name__ == "__main__":
    museum = Museum("datasets/BBDD", rm_frame=True)
    result = museum.compute_similarity("datasets/BBDD/bbdd_00002.jpg")
    print(result)
    #image = cv2.imread(args["image"])
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)