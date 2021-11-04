import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from similarity import compute_similarity as compute_similarity_measure

class SiftDescriptor(object):

    def __init__(self) -> None:
        self.sift_descriptor = cv2.xfeatures2d.SIFT_create()

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

    def compute_descriptor(self, image: np.ndarray, bbox:tuple = None):
        """ input_params
                bbox: tuple containing two tuples, where first tuple is top-left coordinate of the mask
                and second tuple in left-bottom coordinates e.g ((0,0),(250,250)), if bbox is None, it will
                compute the histograms for all the tiles            
        """
        #if the tile size is not defined at init, we will divide the image in 
        mask = self.compute_hist_mask(image, bbox)
        if mask is not None:
            image = cv2.bitwise_and(image, image, mask=mask)
        
        keypoints_2, descriptors_2 = self.sift_descriptor.detectAndCompute(image,None)
        return keypoints_2, descriptors_2

    def compute_image_similarity(self, dataset, similarity_mode, query_img, text_extractor_method):
        result = []
        if text_extractor_method is not None:
            bbox_query ,  _,_= text_extractor_method(query_img,None,None)
        else:
            bbox_query = None
        _, query_img_features = self.compute_descriptor(query_img, bbox_query)
        for image in dataset.keys():
            #image_hist = self.compute_descriptor(dataset[image]["image_obj"])
            sim_result = compute_similarity_measure(dataset[image]["sift_desc"], query_img_features, similarity_mode)
            result.append([image, sim_result])
        return result