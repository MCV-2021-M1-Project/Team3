import os
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np


VALID_IMAGE_FORMATS = ['JPEG']


class FileIsNotImageError(Exception):
    """The provided file does not match an Image type"""
    pass

class Museum(object):
    """
    A class for loading the museum images and 
    extrat its feature descriptors.
    """

    def __init__(self, data_set_path):
        self.image_dataset = self.load_images_dataset(data_set_path)        

    def load_images_dataset(self, path_images):
        """
        Method to load the image from dataset path
        """
        image_dataset = {}
        for image_name in os.listdir(path_images)[:50]:
            if self.file_is_image(os.path.join(path_images, image_name)):
                image = cv2.imread(os.path.join(path_images, image_name))
                hist = self.compute_histogram(image)
                #cv2.imshow("",image)
                #cv2.waitKey(0)
                image_num = self.get_image_number(image_name)
                image_dataset[image_num] = {
                    "image_name": image_name,
                    "image_obj": image
                }
        return image_dataset
    
    def load_query_img(self, image_path):
        if self.file_is_image(image_path):
            image = cv2.imread(image_path)
            return image
        else:
            raise FileIsNotImageError("The provided file does not match an Image type")

    def compute_similarity(self, image_path, simalarity_fn):
        result = []
        query_img = self.load_query_img(image_path)  
        query_img_hist = self.compute_histogram(query_img)
        for image in self.image_dataset.keys():
            image_hist = self.compute_histogram(self.image_dataset[image]["image_obj"])
            sim_result = simalarity_fn(image_hist, query_img_hist, cv2.HISTCMP_BHATTACHARYYA)
            result.append([image, sim_result])
        return result

    def compute_histogram(self, image, color="gray", plot=False):
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
    def get_image_number(image_name):
        """
        Method to get the image number
        image_name(str)
        """
        image_num = image_name.split("_")[1].split(".")[0]
        return int(image_num)

    @staticmethod
    def file_is_image(image_path):
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

museum = Museum("/Users/manelguz/Team3/datasets/BBDD")
result = museum.compute_similarity("/Users/manelguz/Team3/datasets/BBDD/bbdd_00002.jpg", cv2.compareHist)
print(result)
#image = cv2.imread(args["image"])
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)