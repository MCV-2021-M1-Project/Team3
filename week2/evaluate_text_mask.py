import results
import os
from museum import Museum
from text_extraction import Text
from evaluate_masks import intersection_over_union
path_query_set = "datasets/qsd1_w2/"
path_gt = "datasets/qsd1_w2/text_boxes.pkl"

gt = results.ground_truth(path_gt)

list_mask_text_results = []
for square in gt:
    list_mask_text_results.append(square[0][0].tolist() + square[0][2].tolist())
print(list_mask_text_results)
#print("List results")
#print(list_mask_text_results)
idx = 0 
for image in sorted(os.listdir(path_query_set)):
    if Museum.file_is_image(os.path.join(path_query_set, image)):
        text_id = Text()
        print("intersection_over_union img " + str(idx))
        bbox ,  _= text_id.text_extraction(os.path.join(path_query_set, image),None,None)
        print(list_mask_text_results[idx])
        print(intersection_over_union(bbox, list_mask_text_results[idx]))
        idx  = idx + 1

