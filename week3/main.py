import argparse
import os
import argparse
import shutil

import museum
import results
from mapk import mapk
from background_remover import Canvas
from text_extraction import Text
from color_descriptor import ColorDescriptor

CANVAS_TMP_FOLDER = "canvas_tmp_folder"
CANVAS_TMP_FOLDER_CROPPED = "canvas_tmp_folder_cropped"

parser_epilog = \
"""
      Example given an image:\n

      python3 main.py "datasets/BBDD" "datasets/qsd1_w1/" "hellinger_similarity" \ --metric lab_3d -k 1
      
      Example given a quey path:\n
      
      python3 main.py "datasets/BBDD" "datasets/qsd1_w1/" "hellinger_similarity" \  "datasets/qsd1_w1/gt_corresps.pkl" --metric lab_3d -k 1
"""

parser = argparse.ArgumentParser(
    description='Computer similarity between the given images and the museum dataset.',
    epilog=parser_epilog
)

# Add the image path argument
parser.add_argument('museum_images_path',
                       metavar='museum_images_path',
                       type=str,
                       help='The museum images path to be loaded')

parser.add_argument('query_image_path',
                       metavar='query_image_path',
                       type=str,
                       help='The image path or directory of images to be compared to with the museum dataset')

parser.add_argument('similarity',
                       metavar='similarity',
                       type=str,
                       choices=["L1_norm", "L2_norm", "cosine_similarity", "histogram_intersection", "hellinger_similarity"],
                       help='The similaritie measures avaliable to compute the measure')

parser.add_argument('-g', '--ground_truth',
                       metavar='ground_truth',
                       type=str,
                       help='The ground_truth result')

parser.add_argument('-m', '--metric',
                       metavar='metric',
                       type=str,
                       choices=["gray","rgb_1d","rgb_3d","hsv","lab", "lab_3d"],
                       default="gray",
                       help='The color spaces avaliable for the histogram computation')


parser.add_argument('-k', '--number_results',
                       metavar='number_results',
                       type=int,
                       default=1,
                       help='The number of top k elements that best match the given image')

parser.add_argument('-b', '--number_blocks',
                       metavar='number_blocks',
                       type=int,
                       default=3,
                       help='The number of rows and cols to devide image')

parser.add_argument('--remove_back',
                      dest='rm_background', 
                      action='store_true',
                      help='Remove the paint background'
                      )
parser.set_defaults(rm_background=False)

# Parse arguments
args = parser.parse_args()
k = args.number_results

descriptor = ColorDescriptor(color_space=args.metric.split("_")[0], scales=args.number_blocks)
museum_similarity_comparator = museum.Museum(
    args.museum_images_path, descriptor, rm_frame=True, similarity_mode=args.similarity
)
canvas = Canvas()
text_extractor = Text()

query_image_path = args.query_image_path
if args.rm_background:
    CANVAS_TMP_FOLDER = os.path.join(query_image_path,CANVAS_TMP_FOLDER)
    CANVAS_TMP_FOLDER_CROPPED = os.path.join(query_image_path,CANVAS_TMP_FOLDER_CROPPED)
    if os.path.isdir(CANVAS_TMP_FOLDER):
        shutil.rmtree(CANVAS_TMP_FOLDER)
        shutil.rmtree(CANVAS_TMP_FOLDER_CROPPED)
    os.mkdir(CANVAS_TMP_FOLDER)
    os.mkdir(CANVAS_TMP_FOLDER_CROPPED)
    list_of_coords = []
    for image in sorted(os.listdir(query_image_path)):
        is_img = museum.Museum.file_is_image(os.path.join(query_image_path, image))
        if is_img:
            _,x,y,w,h,x2,y2,w2,h2 = canvas.background_remover(os.path.join(query_image_path, image), os.path.join(os.getcwd(), CANVAS_TMP_FOLDER), os.path.join(os.getcwd(), CANVAS_TMP_FOLDER_CROPPED) , image)
            temp_list = [(x,y)]
            if x2 > 0:
                temp_list.append((x2,y2))
            list_of_coords.append(temp_list)
    results.create_results(list_of_coords, file_path=os.path.join(query_image_path,"coordinates_mask_original_image.pkl"))
    query_image_path = CANVAS_TMP_FOLDER_CROPPED

final_result = []
if os.path.isdir(query_image_path):
    for image in sorted(os.listdir(query_image_path)):
        try:
            # working multiscale
            print(os.path.join(query_image_path, image))
            result = museum_similarity_comparator.compute_similarity(
                os.path.join(query_image_path, image), args.metric, 
                text_extractor_method=text_extractor.text_extraction
            )
            # working at given image size
            #result = museum_similarity_comparator.compute_similarity(os.path.join(query_image_path, image), args.metric)
        except museum.FileIsNotImageError:
            continue
        result.sort(key=lambda x: x[1]) # resulting score sorted
        result = result[:k] # take the k elements
        result = [ key for key, val in result] ## For eache element, get only the image and forget about the actual similarity value
        final_result.append(result)
    if args.ground_truth is not None:
        gt = results.ground_truth(args.ground_truth)
        mapk_result = mapk(gt, final_result, k=k)
        print("Resulting Mapk:")
        print(mapk_result)
        print("Ground Truth")
        print(gt)
else:
    # working multiscale
    result = museum_similarity_comparator.compute_similarity(
        query_image_path, args.metric, 
        text_extractor_method=text_extractor.text_extraction
    )
    # working at given image size
    #result = museum_similarity_comparator.compute_similarity(os.path.join(query_image_path, image), args.metric)
    result.sort(key=lambda x: x[1]) # resulting score sorted
    result = result[:k]
    final_result = [ key for key, val in result] ## For eache element, get only the image and forget about the actual similarity value

print("Final Guess")
print(final_result)
results.create_results(final_result)

# For evaluation purposes

# python3 main.py "datasets/BBDD" "datasets/qst1_w1/" "L1_norm" -g "datasets/qst1_w1/gt_corresps.pkl" --metric rgb_3d -k 1

#python3 main.py "datasets/BBDD" "datasets/qst2_w1/" "L1_norm" -g "datasets/qst2_w1/gt_corresps.pkl" --metric rgb_3d -k 1 --remove_back