import argparse
import os
import argparse
import shutil

import museum
import results
from mapk import mapk
from background_remover import Canvas

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

parser.add_argument('--remove_back',
                      dest='rm_background', 
                      action='store_true',
                      help='Remove the paint background'
                      )
parser.set_defaults(rm_background=False)

# Parse arguments
args = parser.parse_args()
k = args.number_results

museum_similarity_comparator = museum.Museum(args.museum_images_path, rm_frame=True, similarity_mode=args.similarity, color_space=args.metric.split("_")[0])
canvas = Canvas()

query_image_path = args.query_image_path
if args.rm_background:
    if os.path.isdir(CANVAS_TMP_FOLDER):
        shutil.rmtree(CANVAS_TMP_FOLDER)
        shutil.rmtree(CANVAS_TMP_FOLDER_CROPPED)
    os.mkdir(CANVAS_TMP_FOLDER)
    os.mkdir(CANVAS_TMP_FOLDER_CROPPED)

    for image in os.listdir(query_image_path):
        is_img = museum.Museum.file_is_image(os.path.join(query_image_path, image))
        if is_img:
            canvas.background_remover(os.path.join(args.query_image_path, image), os.path.join(os.getcwd(), CANVAS_TMP_FOLDER), os.path.join(os.getcwd(), CANVAS_TMP_FOLDER_CROPPED) , image)
    query_image_path = CANVAS_TMP_FOLDER_CROPPED

final_result = []
if os.path.isdir(query_image_path):
    for image in sorted(os.listdir(query_image_path)):
        try:
            result = museum_similarity_comparator.compute_similarity(os.path.join(query_image_path, image), args.metric)
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
    result = museum_similarity_comparator.compute_similarity(query_image_path, args.metric)
    result.sort(key=lambda x: x[1]) # resulting score sorted
    result = result[:k]
    final_result = [ key for key, val in result] ## For eache element, get only the image and forget about the actual similarity value

print("Final Guess")
print(final_result)

# For evaluation purposes

# python3 main.py "datasets/BBDD" "datasets/qst1_w1/" "L1_norm" -g "datasets/qst1_w1/gt_corresps.pkl" --metric rgb_3d -k 1

#python3 main.py "datasets/BBDD" "datasets/qst2_w1/" "L1_norm" -g "datasets/qst2_w1/gt_corresps.pkl" --metric rgb_3d -k 1 --remove_back