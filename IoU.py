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

##Example 
bbox_actual = [39, 63, 203, 112]
bbox_predicted = [54, 66, 198, 114]

IoU = intersection_over_union(bbox_predicted,bbox_actual)
print(IoU)