import results
import os
from museum import Museum
from text_extraction import Text
from evaluate_masks import intersection_over_union
import pickle
path_query_set = "datasets/qst1_w2/"
"""
path_gt = "datasets/qsd1_w2/text_boxes.pkl"

gt = results.ground_truth(path_gt)

list_mask_text_results = []
for square in gt:
    list_mask_text_results.append(square[0][0].tolist() + square[0][2].tolist())
print(list_mask_text_results)
"""
#print("List results")
#print(list_mask_text_results)
"""
idx = 0 
result =[]
for image in sorted(os.listdir(path_query_set)):
    if Museum.file_is_image(os.path.join(path_query_set, image)):
        text_id = Text()
        #print("intersection_over_union img " + str(idx))
        bbox ,  _= text_id.text_extraction(os.path.join(path_query_set, image),None,None)
        #print(list_mask_text_results[idx])
        #print(intersection_over_union(bbox, list_mask_text_results[idx]))
        #idx  = idx + 1
        result.append(bbox)

print(result)
results.create_results(result, os.path.join(path_query_set,"text_boxes.pkl"))

"""
path_query_set_cropped = "datasets/qst2_w2/canvas_tmp_folder_cropped"
path_query_set = "datasets/qst2_w2/test"
result =[]

with open("datasets/qst2_w2/coordinates_mask_original_image.pkl","rb") as file_handle:
    image_coords = pickle.load(file_handle)
print(len(image_coords))
print(image_coords)    
for image,coords in zip(sorted(os.listdir(path_query_set)),image_coords):
    print(image)
    if Museum.file_is_image(os.path.join(path_query_set, image)):
        print(image)
        partial = []
        cropped_img = os.path.join(path_query_set_cropped, "crop_{}.jpg".format(image.split(".")[0]))
        if os.path.isfile(cropped_img):
            text_id = Text()
            #print("intersection_over_union img " + str(idx))
            bbox ,  _= text_id.text_extraction(os.path.join(path_query_set, image),None,None)
            x, y = coords[0]
            print(coords)
            bbox = [bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y] 
            partial.append(bbox)
        cropped_img = os.path.join(path_query_set_cropped, "crop_{}_2.jpg".format(image.split(".")[0]))
        if os.path.isfile(cropped_img):
            text_id = Text()
            #print("intersection_over_union img " + str(idx))
            bbox ,  _= text_id.text_extraction(os.path.join(path_query_set, image),None,None)
            print("original")
            print(bbox)
            x, y = coords[1]
            bbox = [bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y] 
            print("tunned")
            print(bbox)
            partial.append(bbox)
        #print(list_mask_text_results[idx])
        #print(intersection_over_union(bbox, list_mask_text_results[idx]))
        #idx  = idx + 1
        #print(partial)
        result.append(partial)
    else:
        print("is not file")

print(len(result))
results.create_results(result, os.path.join(path_query_set,"text_boxes.pkl"))
