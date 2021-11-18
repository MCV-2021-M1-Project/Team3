import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from similarity import compute_similarity as compute_similarity_measure
from keypoint_similarities import KeypointSimilarity
class KeypointDescriptor(object):

    def __init__(self,descriptor_type:str = 'ORB') -> None:
        self.descriptors = {
            ##'sift': cv2.xfeatures2d.sift_create(),
            'surf': cv2.ORB_create(1000),
            'sift' : cv2.xfeatures2d.SIFT_create(nfeatures=400),
            'ORB':cv2.ORB_create(1000)
        }
        self.descriptor_type = descriptor_type
        self.similarity = KeypointSimilarity()
        #self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)        
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
        #mask = self.compute_hist_mask(image, bbox)
        #if mask is not None:
        #    image = cv2.bitwise_and(image, image, mask=mask)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        image = cv2.resize(image, dsize=(500,500))
        mask = np.ones(image.shape[:2], dtype=np.uint8)*255
        if bbox is not None:
            cv2.rectangle(mask, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0), thickness = -1)        
            keypoints_2, descriptors_2 = self.descriptors[self.descriptor_type].detectAndCompute(image,mask)
        else:
            keypoints_2, descriptors_2 = self.descriptors[self.descriptor_type].detectAndCompute(image,None)
        return keypoints_2, descriptors_2
    def compute_similarity(self,image_descriptor,query_image_descriptor):
        matches = self.matcher.match(image_descriptor,query_image_descriptor)
        matches = sorted(matches, key = lambda x:x.distance)
        #print(matches[0].distance)
        #print(len([match for match in matches if match.distance < 0.5]))
        return matches,len([match for match in matches if match.distance < 100])
        """if len(matches) == 0:
            return len(matches)
        #print(len(matches))
        total_distance = 0
        try:
            for i in range(self.top_matches):
                total_distance += matches[i].distance
        except:
            #return total_distance/(i+1) 
            #        
        #return total_distance/self.top_matches"""    



    def compute_image_similarity(self, dataset, similarity_mode, query_img, text_extractor_method):
        result = []
        if text_extractor_method is not None:
            bbox_query , _, _, _, _= text_extractor_method(query_img,None,None)
            
        else:
            bbox_query = None
            print('bbox query is none')
        keypoints, query_img_features = self.compute_descriptor(query_img, None)
        for image in dataset.keys():

            sim_result = self.similarity.match_keypoints_descriptors(dataset[image]["descriptor"], query_img_features)

            #if sim_result[0] != 10000:
                #img3 = cv2.drawMatchesKnn(dataset[image]['image_obj'],keypoints_dataset,query_img,keypoints,sim_result[1],None, flags=2)                
                #print(cv2.imwrite('/home/marcelo/Documents/Master_CV/M1/'+dataset[image]["image_name"],img3))
            result.append([image, sim_result[0]])
        return result