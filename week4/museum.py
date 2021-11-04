import os
from PIL import Image, ImageChops
from numpy.lib.arraysetops import isin
import cv2
from text_extraction import Text
from matplotlib import pyplot as plt
import numpy as np
from color_descriptor import ColorDescriptor
from noise_remover import NoiseRemover
from texture_descriptor import TextureDescriptor
from keypoint_descriptor import KeypointDescriptor
VALID_IMAGE_FORMATS = ['JPEG']


class FileIsNotImageError(Exception):
    """The provided file does not match an Image type"""
    pass

class Museum(object):
    """
    A class for loading the museum images and 
    extrat its feature descriptors.
    """

    def __init__(self, data_set_path: str, descriptor:callable, similarity_mode: list, rm_frame:bool = False, rm_noise:bool = False):
        """[summary]

        Args:
            data_set_path (str): museum dataset path
            descriptor (callable): Instance of the class of the descriptor of to be used ALLREADY INSTANCIATED (color, text, texture...)
            similarity_mode (str, optional): [description]. Defaults to "L1_norm".
            rm_frame (bool, optional): [description]. Defaults to False.
        """
        self.rm_frame = rm_frame
        self.descriptor = descriptor
        self.image_dataset = self.load_images_dataset(data_set_path)
        self.similarity_mode = similarity_mode        
        if  not len(self.similarity_mode) == len(self.descriptor):
            print('number of descriptors and number of similarities must be the same')
        self.rm_noise = rm_noise
        self.noise_remover = NoiseRemover()

    def compute_descriptor_for_dataset(self, dataset, image):
        for descriptor in self.descriptor:
            if isinstance(descriptor,ColorDescriptor):
                dataset[image]["color_desc"] = descriptor.compute_descriptor(
                    dataset[image]["image_obj"]
                )
            elif isinstance(descriptor,TextureDescriptor):
                dataset[image]["texture_desc"] = descriptor.compute_descriptor(
                    dataset[image]["image_obj"]
                )
            elif isinstance(descriptor,KeypointDescriptor):
                dataset[image]["keypoints"], dataset[image]["descriptor"] = descriptor.compute_descriptor(
                    dataset[image]["image_obj"]
                )

    def extract_text_from_files(self, text_path):
        #print(text_path)
        with open(text_path, "r") as file_r:
            text = file_r.read()
        return text

    def load_images_dataset(self, image_path: str, ):
        """
        Method to load the image from dataset path
        """
        image_dataset = {}
        for image_name in os.listdir(image_path):
            if self.file_is_image(os.path.join(image_path, image_name)):
                image = cv2.imread(os.path.join(image_path, image_name))
                image_text = self.extract_text_from_files(os.path.join(image_path, image_name.replace(".jpg", ".txt")))
                if self.rm_frame:
                    image = self.remove_frame(image)
                image_num = self.get_image_number(image_name)
                image_dataset[image_num] = {
                    "image_name": image_name,
                    "image_obj": image,
                    "image_text": image_text
                }
                self.compute_descriptor_for_dataset(image_dataset, image_num)
        return image_dataset
    
    def load_query_img(self, image_path: str):
        print(image_path)
        if self.file_is_image(image_path):
            image = cv2.imread(image_path)
            if self.rm_frame:
                image = self.remove_frame(image)
            return image
        else:
            raise FileIsNotImageError("The provided file does not match an Image type")
    
    def compute_similarity(self, image_set:str, text_extractor_method:callable):
        set_result = []
        print(image_set)
        if os.path.isdir(image_set):
            for image in os.listdir(image_set):
                #try:
                    query_img = self.load_query_img(os.path.join(image_set,image))
                    print(image)
                    if self.rm_noise:
                        img_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY) 
                        if  self.noise_remover.is_noisy_img(img_gray):
                            denoised_img = self.noise_remover.remove_noise(query_img, "median", 3)
                            query_img = denoised_img
                    if  not isinstance(self.descriptor, list):      
                        set_result.append(
                        self.descriptor.compute_image_similarity(
                            self.image_dataset, self.similarity_mode[0],
                            query_img, text_extractor_method
                        )
                    )
                    else:
                        similarities = []
                        for i,descriptor in enumerate(self.descriptor):                            
                            similarities.append(descriptor.compute_image_similarity(
                                    self.image_dataset,self.similarity_mode[i],
                                    query_img, text_extractor_method
                                ))

                        set_result.extend([sim for sim in similarities])   

                      
                        
                     


                #except FileIsNotImageError:
                #    pass
        else:
            query_img = self.load_query_img(image_set) 
            """set_result = self.descriptor.compute_image_similarity(
                self.image_dataset, self.similarity_mode, 
                query_img, text_extractor_method
            )"""
            if  not isinstance(self.descriptor, list):      
                set_result = self.descriptor.compute_image_similarity(
                self.image_dataset, self.similarity_mode[0], 
                query_img, text_extractor_method
            )
                    
                return set_result    
            else:
                similarities = []
                for i,descriptor in enumerate(self.descriptor):

                    similarities.append(descriptor.compute_image_similarity(
                                    self.image_dataset,self.similarity_mode[i],
                                    query_img, text_extractor_method
                                ))
                
                return similarities                
                           
        


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