import os
from PIL import Image, ImageChops
import cv2
from matplotlib import pyplot as plt
import numpy as np
from color_descriptor import ColorDescriptor
VALID_IMAGE_FORMATS = ['JPEG']


class FileIsNotImageError(Exception):
    """The provided file does not match an Image type"""
    pass

class Museum(object):
    """
    A class for loading the museum images and 
    extrat its feature descriptors.
    """

    def __init__(self, data_set_path: str, descriptor:callable, similarity_mode: str ="L1_norm", rm_frame:bool = False):
        """[summary]

        Args:
            data_set_path (str): museum dataset path
            descriptor (callable): Instance of the class of the descriptor of to be used ALLREADY INSTANCIATED (color, text, texture...)
            similarity_mode (str, optional): [description]. Defaults to "L1_norm".
            rm_frame (bool, optional): [description]. Defaults to False.
        """
        self.rm_frame = rm_frame
        self.image_dataset = self.load_images_dataset(data_set_path)
        self.similarity_mode = similarity_mode
        self.descriptor = descriptor

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
    
    def compute_similarity(self, image_set: str, metric:str, text_extractor_method:callable):
        set_result = []
        if os.path.isdir(image_set):
            for image in os.listdir(image_set):
                try:
                    query_img = self.load_query_img(os.path.join(image_set,image)) 
                    set_result.append(
                        self.descriptor.compute_image_similarity(
                            self.image_dataset, self.similarity_mode,
                            query_img, metric, text_extractor_method
                        )
                    )
                    print(set_result)
                except FileIsNotImageError:
                    pass
        else:
            query_img = self.load_query_img(image_set) 
            set_result = self.descriptor.compute_image_similarity(
                self.image_dataset, self.similarity_mode, 
                query_img, metric, text_extractor_method
            )
            print(set_result)
        return set_result


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
    descriptor = ColorDescriptor(color_space="lab_3d", scales= 3)
    museum = Museum("datasets/BBDD",descriptor, rm_frame=True)
    result = museum.compute_similarity("datasets/BBDD/bbdd_00002.jpg")
    print(result)
    #image = cv2.imread(args["image"])
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)