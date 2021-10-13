import os
from PIL import Image, ImageChops
import cv2
from matplotlib import pyplot as plt
import numpy as np
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

    def __init__(self, data_set_path: str, similarity_mode: str ="L1_norm", color_space:str ="gray", rm_frame:bool = False):
        self.rm_frame = rm_frame
        self.image_dataset = self.load_images_dataset(data_set_path)
        self.similarity_mode = similarity_mode
        self.color_space = color_space

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
    
    def compute_similarity(self, image_set: str):
        set_result = []
        if os.path.isdir(image_set):
            for image in os.listdir(image_set):
                try:
                    set_result.append(self.compute_image_similarity(os.path.join(image_set,image)))
                except FileIsNotImageError:
                    pass
        else:
            set_result = self.compute_image_similarity(image_set)
        return set_result

    def compute_image_similarity(self, image_path: str):
        result = []
        query_img = self.load_query_img(image_path)  
        query_img_hist = self.compute_histogram(query_img)
        for image in self.image_dataset.keys():
            image_hist = self.compute_histogram(self.image_dataset[image]["image_obj"])
            sim_result = compute_similarity(image_hist, query_img_hist,self.similarity_mode)
            result.append([image, sim_result])
        return result

    def compute_histogram(self, image: np.ndarray, plot=False):
        """
        Method to compute the histogram of an image
        """
        color_space = {
            "gray": cv2.COLOR_BGR2GRAY,
            "rgb": cv2.COLOR_BGR2RGB, 
            "hsv": cv2.COLOR_BGR2HSV,
            "lab": cv2.COLOR_BGR2LAB
        }

        image = cv2.cvtColor(image, color_space[self.color_space])
        
        chans = cv2.split(image) 
        if self.color_space != "hsv":
            hist = self.compute_standard_histogram(chans)
        else:
            hist = self.compute_hsv_histogram(chans)

        #hist = hist.astype(np.uint8)
        if plot:
            plt.figure()
            plt.title("Histogram")
            plt.xlabel("Bins")
            plt.ylabel("# of Pixels")
            for color_hist in np.split(hist, len(chans)):
                plt.plot(color_hist)
            plt.xlim([0, 256])
            plt.show()
        return hist

    def compute_standard_histogram(self, channels: np.ndarray):
        hist_chan = []
        for channel in channels:
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist)
            hist_chan.append(hist)
        hist = np.concatenate((hist_chan))
        return hist

    def compute_hsv_histogram(self, channels: np.ndarray):
        # reduce the bins for brightness channel
        hist_chan = []
        max_bins = 256
        for channel, bins, max_val in zip(channels,[int(180/(256/max_bins)), max_bins, int(max_bins/8)],[180,256,256]):
            hist = cv2.calcHist([channel],[0], None, [bins], [0, max_val])
            hist = cv2.normalize(hist, hist)
            hist_chan.append(hist)
        hist = np.concatenate((hist_chan))
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