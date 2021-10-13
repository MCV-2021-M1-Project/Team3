import numpy as np
import cv2



def compute_similarity(vector_1: np.uint8, vector_2: np.uint8,mode:str) -> float:

    func = switcher.get(
        mode, lambda: "Invalid similarity function, pick one between: L1_norm,L2_norm,cosine_similairty,histogram_intersection,hellinger_similarity")
    return func(vector_1, vector_2)


def L1_norm(vector_1: np.uint8, vector_2: np.uint8) -> np.float32:
    return np.linalg.norm(vector_1 - vector_2,1)


def L2_norm(vector_1: np.uint8, vector_2: np.uint8) -> np.float32:
    return np.linalg.norm(vector_1 - vector_2,2)


def cosine_similarity(vector_1: np.  uint8, vector_2: np.uint8) -> np.float32:
    return np.dot(vector_1.reshape(-1), vector_2.reshape(-1))/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2))


def histogram_intersection(vector_1: np.uint8, vector_2: np.uint8) -> np.float32:
    #return np.sum(np.minimum(vector_1, vector_2))
    return  np.true_divide(np.sum(np.minimum(vector_1, vector_2)), np.sum(vector_2))


def hellinger_similarity(vector_1: np.uint8, vector_2: np.uint8) -> np.float32:
    return np.sqrt(np.sum((np.sqrt(vector_1) - np.sqrt(vector_2)) ** 2)) / np.sqrt(2)

switcher = {
    "L1_norm": L1_norm,
    "L2_norm": L2_norm,
    "cosine_similarity": cosine_similarity,
    "histogram_intersection": histogram_intersection,
    "hellinger_similarity": hellinger_similarity,
}

if __name__ == "__main__":
    #img1 = cv2.imread("./datasets/qsd1_w2/00000.jpg")
    #img2 = cv2.imread("./datasets/qsd1_w2/00000.jpg")
    x = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0]) 
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    for item in switcher.keys():
        similarity_score = compute_similarity(x,y,item)
        print("similarity with {} is {} with dtype {}".format(item,similarity_score,similarity_score.dtype))