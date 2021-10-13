import cv2 as cv
import numpy as np
import os

def generate_binary_mask(mask):
    if len(mask.shape) == 3:
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    if not(np.unique(mask) == (0, 1)).all():
        mask = (mask == 255).astype(float)

    return mask

def evaluate_mask(og_mask: np.ndarray, gen_mask: np.ndarray):
    
    og_mask, gen_mask = generate_binary_mask(og_mask), generate_binary_mask(gen_mask)
    assert (og_mask.shape == gen_mask.shape)
    
    og_mask = og_mask.flatten()
    gen_mask = gen_mask.flatten()
    
    tp = int(np.sum(og_mask * gen_mask)) #True Positives (tp)
    tn = int(np.sum((1 - og_mask) * (1 - gen_mask))) #True Negatives (tn)
    fn = int(np.sum((1 - og_mask) * gen_mask)) #False Negatives (fn)
    fp = int(np.sum(og_mask * (1 - gen_mask))) #False Positives (fp)
  
    try: precision = tp / (tp + fp)
    except ZeroDivisionError:
        recall = 0

    try: recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1, tp, fp, fn, tn

if __name__ == '__main__':

    valid_images = [".png"]
    load_directory = 'C:\\Users\\usuario\\Documents\\GitHub\\ABC\\CV_M1\\W1\\QSD2\\'
    save_direcory = 'C:\\Users\\usuario\\Documents\\GitHub\\ABC\\CV_M1\\W1\\QSD2\\generated_masks\\'

    for f in os.listdir(load_directory):
        file_name = typex = os.path.splitext(f)[0]
        typex = os.path.splitext(f)[1]
        if typex.lower() not in valid_images:
            continue
        print('Calculated precision img ',f)
        og_mask = cv.imread(load_directory+f)
        gen_mask = cv.imread(save_direcory+'mask_'+f)
        precision, recall, f1, tp, fp, fn, tn = evaluate_mask(og_mask,gen_mask)
        print('Precision: ',precision,'Recall: ', recall,'F1: ' ,f1,'TP :', tp,'FP: ', fp,'FN: ', fn, 'TN: ', tn)

