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
    IoU = tp / np.sum(np.logical_or(og_mask, gen_mask))
  
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
    
    return precision, recall, f1, tp, fp, fn, tn, IoU

def intersection_over_union(bbox_predicted, bbox_gt):
	"""
    Given two bounding boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.
    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
    
    Returns:
        (float) The Intersect of Union score.
    
    """
    
    # determine the (x, y) coordinates of the intersection rectangle
	xA = max(bbox_predicted[0], bbox_gt[0])
	yA = max(bbox_predicted[1], bbox_gt[1])
	xB = min(bbox_predicted[2], bbox_gt[2])
	yB = min(bbox_predicted[3], bbox_gt[3])
	
    # compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth rectangles
	bbox_predicted_Area = (bbox_predicted[2] - bbox_predicted[0] + 1) * (bbox_predicted[3] - bbox_predicted[1] + 1)
	bbox_gt_Area = (bbox_gt[2] - bbox_gt[0] + 1) * (bbox_gt[3] - bbox_gt[1] + 1)
	
    # compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the interesection area
	iou = interArea / float(bbox_predicted_Area + bbox_gt_Area - interArea)
	
    # return the intersection over union value
	return iou


if __name__ == '__main__':

    valid_images = [".png"]
    load_directory = 'C:\\Users\\JQ\\Documents\\GitHub\\ABC\\CV_M1\\W1\\QSD2\\'
    save_direcory = 'C:\\Users\\JQ\\Documents\\GitHub\\ABC\\CV_M1\\W1\\QSD2\\generated_masks\\'

    for f in os.listdir(load_directory):
        file_name = typex = os.path.splitext(f)[0]
        typex = os.path.splitext(f)[1]
        if typex.lower() not in valid_images:
            continue
        print('Calculated precision img ',f)
        og_mask = cv.imread(load_directory+f)
        gen_mask = cv.imread(save_direcory+f)
        precision, recall, f1, tp, fp, fn, tn, Iou = evaluate_mask(og_mask,gen_mask)
        print('Precision: ',precision,'Recall: ', recall,'F1: ' ,f1,'TP :', tp,'FP: ', fp,'FN: ', fn, 'TN: ', tn, 'IoU:', Iou)

    ##Example 
    """
    bbox_actual = [39, 63, 203, 112]
    bbox_predicted = [54, 66, 198, 114]

    IoU = intersection_over_union(bbox_predicted,bbox_actual)
    print(IoU)
    """
