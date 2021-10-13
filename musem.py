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

    def __init__(self, data_set_path: str, rm_frame:bool = False,similarity_mode ="L1_norm"):
        self.rm_frame = rm_frame
        self.image_dataset = self.load_images_dataset(data_set_path)
        self.similarity_mode = similarity_mode

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

    def compute_similarity(self, image_path: str):
        result = []
        query_img = self.load_query_img(image_path)  
        query_img_hist = self.compute_histogram(query_img, color="gray")
        for image in self.image_dataset.keys():
            image_hist = self.compute_histogram(self.image_dataset[image]["image_obj"], color="gray")
            sim_result = compute_similarity(image_hist, query_img_hist,self.similarity_mode)
            result.append([image, sim_result])
        return result

    def compute_histogram(self, image: np.ndarray , color="gray", plot=False):
        """
        Method to compute the histogram of an image
        """
        color_space = {
            "gray": cv2.COLOR_BGR2GRAY,
            "rgb": cv2.COLOR_BGR2RGB, 
            "hsv": cv2.COLOR_BGR2HSV,
            "lab": cv2.COLOR_BGR2LAB
        }
        image = cv2.cvtColor(image, color_space[color])
        
        chans = cv2.split(image) 
        hist_chan = []
        for channel in chans:
            hist_chan.append(cv2.calcHist([channel], [0], None, [256], [0, 256]))
        hist = np.concatenate((hist_chan))

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
        Method to remove the background given a fixed number of pixels
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


museum = Museum("datasets/BBDD", rm_frame=True)
result = museum.compute_similarity("datasets/BBDD/bbdd_00002.jpg")
print(result)
#image = cv2.imread(args["image"])
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)