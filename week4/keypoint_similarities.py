import numpy as np
import cv2


class KeypointSimilarity(object):
    def __init__(self, method: str = 'flann', distance: str = 'l1') -> None:

        self.distances = {
            'hamming': cv2.NORM_HAMMING,
            'hamming2': cv2.NORM_HAMMING2,
            'l1': cv2.NORM_L1,
            'l2': cv2.NORM_L2,

        }
        self.matches_limit = 20
        self.method = method
        self.distance = distance

        self.matching = {
            'bruteforce': self.bruteforce_matching,
            'flann': self.flann_matching,
        }

    def lowe_filter(self, matches, k=0.8):
        filtered = []
        for m, n in matches:
            if m.distance < k * n.distance:
                filtered.append(m)
        return filtered

    def keypoints_based_similarity(self, matches, min_matches=6, max_dist=800):
        # print(len(matches))
        if len(matches) > self.matches_limit:
            return len(matches)
        else:
            return 0

    def bruteforce_matching(self, vector_1, vector_2, distance):
        bf = cv2.BFMatcher(normType=distance, crossCheck=False)
        matches = bf.knnMatch(vector_1, vector_2, k=5)
        return self.keypoints_based_similarity(self.lowe_filter(matches))

    def flann_matching(self, vector_1, vector_2, distance):
        flann = cv2.FlannBasedMatcher(
            {'algorithm': 0, 'trees': 5}, {'checks': 50})
        matches = flann.knnMatch(vector_1, vector_2, k=2)
        return self.keypoints_based_similarity(self.lowe_filter(matches))

    def match_keypoints_descriptors(self, vector_1, vector_2, method="flann", distance="l2"):
        if vector_1 is None or vector_2 is None or len(vector_2) <= 1 or len(vector_1) <= 1:
            return 0
        return self.matching[self.method](vector_1.astype(np.float32), vector_2.astype(np.float32), self.distances[self.distance])
