import results
import os
from museum import Museum
from text_extraction import Text
#from evaluate_masks import intersection_over_union
import argparse
import pickle
import shutil
from background_remover import Canvas
import glob

CANVAS_TMP_FOLDER = "canvas_tmp_folder"
CANVAS_TMP_FOLDER_CROPPED = "canvas_tmp_folder_cropped"
DEBUG_TEXT_MASK_IMG_FOLDER = "mask_image"

parser = argparse.ArgumentParser(
    description='Computer similarity between the given images and the museum dataset.',
)

parser.add_argument('path_query_set',
                       metavar='path_query_set',
                       type=str,
                       help='The image path or directory of images to be compared to with the museum dataset')

parser.add_argument('-t', '--test_set',
                       metavar='test_set',
                       type=int,
                       choices=[1,2],
                       help='kind of test set')

parser.add_argument('-d', '--debug',
                       metavar='debug',
                       type=bool,
                       default=False,
                       help='save mask img of test set')

# Parse arguments
args = parser.parse_args()

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
path_query_set = args.path_query_set
test_set = args.test_set

if args.debug:
    if os.path.isdir(DEBUG_TEXT_MASK_IMG_FOLDER):
        shutil.rmtree(DEBUG_TEXT_MASK_IMG_FOLDER)
    os.mkdir(DEBUG_TEXT_MASK_IMG_FOLDER)  

if test_set == 1:
    idx = 0 
    result =[]
    for image in sorted(os.listdir(path_query_set)):
        if Museum.file_is_image(os.path.join(path_query_set, image)):
            print(image)
            text_id = Text()
            #print("intersection_over_union img " + str(idx))
            img = text_id.input_image(os.path.join(path_query_set, image))
            if args.debug:     
                bbox ,  _, _= text_id.text_extraction(img,DEBUG_TEXT_MASK_IMG_FOLDER,image.split(".")[0])
            else:
                bbox ,  _, _= text_id.text_extraction(img,None,None)
            #print(list_mask_text_results[idx])
            #print(intersection_over_union(bbox, list_mask_text_results[idx]))
            #idx  = idx + 1
            result.append(bbox)

    print(result)
    results.create_results(result, os.path.join(path_query_set,"text_boxes.pkl"))

elif test_set == 2:
    canvas = Canvas()


    CANVAS_TMP_FOLDER = os.path.join(path_query_set,CANVAS_TMP_FOLDER)
    CANVAS_TMP_FOLDER_CROPPED = os.path.join(path_query_set,CANVAS_TMP_FOLDER_CROPPED)
    if os.path.isdir(CANVAS_TMP_FOLDER):
        shutil.rmtree(CANVAS_TMP_FOLDER)
        shutil.rmtree(CANVAS_TMP_FOLDER_CROPPED)
    os.mkdir(CANVAS_TMP_FOLDER)
    os.mkdir(CANVAS_TMP_FOLDER_CROPPED)
    list_of_coords = []
    for image in sorted(os.listdir(path_query_set)):
        if Museum.file_is_image(os.path.join(path_query_set, image)):
            _,x,y,w,h,x2,y2,w2,h2 = canvas.background_remover(os.path.join(path_query_set, image), os.path.join(os.getcwd(), CANVAS_TMP_FOLDER), os.path.join(os.getcwd(), CANVAS_TMP_FOLDER_CROPPED) , image)
            temp_list = [(x,y)]
            if x2 > 0:
                temp_list.append((x2,y2))
            list_of_coords.append(temp_list)
    results.create_results(list_of_coords, file_path=os.path.join(path_query_set,"coordinates_mask_original_image.pkl"))

    path_query_set_cropped = os.path.join(path_query_set,"canvas_tmp_folder_cropped")
    #path_query_set = "datasets/qst2_w2/test"
    result =[]

    coordinates_mask_original_file = os.path.join(path_query_set,"coordinates_mask_original_image.pkl")
    with open(coordinates_mask_original_file,"rb") as file_handle:
        image_coords = pickle.load(file_handle)
    for image,coords in zip(sorted(glob.glob(os.path.join(path_query_set, "*.jpg"),recursive=False)),image_coords):
        image = image.split("/")[-1]
        if Museum.file_is_image(os.path.join(path_query_set, image)):
            print(image)
            partial = []
            cropped_img = os.path.join(path_query_set_cropped, "crop_{}.jpg".format(image.split(".")[0]))
            if os.path.isfile(cropped_img):
                text_id = Text()
                #print("intersection_over_union img " + str(idx))
                img = text_id.input_image(cropped_img)  
                print(cropped_img)
                if args.debug:    
                    bbox ,  _, _= text_id.text_extraction(img,DEBUG_TEXT_MASK_IMG_FOLDER,image.split(".")[0])
                else:
                    bbox ,  _, _= text_id.text_extraction(img,None,None)
                x, y = coords[0]
                bbox = [bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y]
                print("Finall box coordinates")
                print(bbox)
                partial.append(bbox)
            cropped_img = os.path.join(path_query_set_cropped, "crop_{}_2.jpg".format(image.split(".")[0]))
            if os.path.isfile(cropped_img):
                text_id = Text()
                #print("intersection_over_union img " + str(idx))
                print(cropped_img)
                img = text_id.input_image(cropped_img)
                if args.debug:    
                    bbox ,  _, _= text_id.text_extraction(img, DEBUG_TEXT_MASK_IMG_FOLDER, image.split(".")[0]+"_2")
                else:
                    bbox ,  _, _= text_id.text_extraction(img,None,None)
                x, y = coords[1]
                bbox = [bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y] 
                print("Finall box coordinates")
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


"""
for qst1_w3
python3 week3/evaluate_text_mask.py datasets/qst1_w3 -t 1

for qst2_w3
python3 week3/evaluate_text_mask.py datasets/qst2_w3 -t 2
"""

