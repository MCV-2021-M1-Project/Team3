from re import match
import numpy as np
import cv2


class KeypointSimilarity(object):
    def __init__(self, method: str = 'bruteforce', distance: str = 'hamming') -> None:

        self.distances = {
            'hamming': cv2.NORM_HAMMING,
            'hamming2': cv2.NORM_HAMMING2,
            'l1': cv2.NORM_L1,
            'l2': cv2.NORM_L2,

        }
        self.matches_limit = 3 # min image matches between the query image and an image in the dataset
        self.method = method
        self.distance = distance

        self.matching = {
            'bruteforce': self.bruteforce_matching,
            'flann': self.flann_matching,
        }

    def lowe_filter(self, matches, k=0.68):
        filtered = []
        for m, n in matches:
            if m.distance < k * n.distance:
                filtered.append([m,n])
        #return matches        
        return filtered

    def keypoints_based_similarity(self, matches):
        #print(len(matches),'len of matches')
        if len(matches) > self.matches_limit:
            return [1/len(matches),matches]
        else:
            return [10000,[]]

    def bruteforce_matching(self, vector_1, vector_2, distance):
        #bf = cv2.BFMatcher(normType=distance,crossCheck=True)
        bf = cv2.BFMatcher(normType=distance)
        matches = bf.knnMatch(vector_1, vector_2,k=2)
        #matches = sorted(matches, key = lambda x:x.distance)
        #return [1/len(matches),matches]
        return self.keypoints_based_similarity(self.lowe_filter(matches))

    def flann_matching(self, vector_1, vector_2, distance):
        index = dict(algorithm=0, trees=5)
        search = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index, search)
        matches = flann.knnMatch(vector_1, vector_2, k=2)
        matches = sorted(matches, key = lambda x:x.distance)
        #print(self.keypoints_based_similarity(self.lowe_filter(matches)))
        return self.keypoints_based_similarity(self.lowe_filter(matches))

    def match_keypoints_descriptors(self, vector_1, vector_2, distance="l2"):
        #print(len(vector_1),len(vector_2),'len of vectors')
        if vector_1 is None or vector_2 is None or len(vector_2) <= 1 or len(vector_1) <= 1:
            return [10000]
        return self.matching[self.method](vector_1, vector_2, self.distances[self.distance])
