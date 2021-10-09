import numpy as np
import cv2
def compute_similarity(image_1:np.uint8,image_2:np.uint8,type:string) -> float:
    
    return




def L1_norm(vector_1:np.uint8,vector_2:np.uint8) -> np.float32:
    return np.linalg.norm(image_1 - image_2,n=1)

def L2_norm(vector_1:np.uint8,vector_2:np.uint8) -> np.float32:
    return np.linalg.norm(image_1 - image_2,n=2)

def cosine_similarity(vector_1:np.  uint8,vector_2:np.uint8) -> np.float32:
    return np.dot(vector_1, vector_2)/np.linalg.norm(vector_1)*np.linalg.norm(vector_2)

def histogram_intersection(vector_1:np.uint8,vector_2:np.uint8) -> np.float32:
    return np.sum(np.minimum(vector_1,vector_2))
    
def hellinger_similarity(vector_1:np.uint8,vector_2:np.uint8) -> np.float32:
    return np.sqrt(np.sum((np.sqrt(vector_1) - np.sqrt(vector_2)) ** 2)) / np.sqrt(2)

if __name__ == "__main__":
    img1 = cv2.imread()