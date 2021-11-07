import argparse
import os
import argparse
import shutil
import numpy as np
import museum
import results
from mapk import mapk
from background_remover import Canvas
from text_remover import Text
from color_descriptor import ColorDescriptor
from texture_descriptor import TextureDescriptor

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


def weight_results_and_normalize_metrics(results, weights,normalize=True):
    assert len(weights) == len(results)
    for i in range(len(weights)):
        results[i].sort(key=lambda x: x[0])    
    results = np.array(results)        
    if normalize:
        for i in range(1,results.shape[2]):            
            results[:,:,i] = (results[:,:,i] - np.min(results[:,:,i]))/np.ptp(results[:,:,i])            
        pass


    
    final_results = []
    for i in range(len(results[0])):
        #print(np.sum(results[:, i, 1]*weights))
        final_results.append([i, np.sum(results[:, i, 1]*weights)])
    return final_results


# Add the image path argument
parser.add_argument('museum_images_path',
                    metavar='museum_images_path',
                    type=str,
                    help='The museum images path to be loaded')

parser.add_argument('query_image_path',
                    metavar='query_image_path',
                    type=str,
                    help='The image path or directory of images to be compared to with the museum dataset')

parser.add_argument('--descriptor',
                    metavar='descriptor',
                    type=str,
                    choices=["color", "texture", "text", "mix", "mix_color_text",
                             "mix_color_texture", "mix_text_texture", "mix_text_texture_color"],
                    help='The similaritie measures avaliable to compute the measure')


"""parser.add_argument('similarity',
                    metavar='similarity',
                    type=str,
                    choices=["L1_norm", "L2_norm", "cosine_similarity",
                             "histogram_intersection", "hellinger_similarity", "levenshtein"],
                    help='The similaritie measures avaliable to compute the measure')"""
parser.add_argument('--similarity', nargs='+', dest='similarity', default=['hellinger_similarity','levenshtein'])                    

parser.add_argument('-g', '--ground_truth',
                    metavar='ground_truth',
                    type=str,
                    help='The ground_truth result')

parser.add_argument('-m', '--metric',
                    metavar='metric',
                    type=str,
                    choices=["gray", "rgb_1d", "rgb_3d",
                             "hsv", "lab", "lab_3d", "HOG"],
                    default="rgb_3d",
                    help='The color spaces avaliable for the histogram computation')
parser.add_argument('-d', '--descriptor_texture_type',
                    metavar='descriptor_texture_type',
                    type=str,
                    choices=["DCT", "LBP", "HOG"],
                    default="DCT",
                    help='descriptor texture type')


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

parser.add_argument('--remove_noise',
                    dest='rm_noise',
                    action='store_true',
                    help='Remove the noise from images'
                    )
parser.add_argument('--weights', nargs='+', dest='weights', default=[1])
parser.set_defaults(rm_noise=False)

# Parse arguments
args = parser.parse_args()
k = args.number_results
descriptor_color = ColorDescriptor(color_space=args.metric.split(
    "_")[0], metric=args.metric, scales=args.number_blocks)
descriptor_texture = TextureDescriptor(color_space=args.metric.split(
    "_")[0], descriptor=args.descriptor_texture_type, scales=args.number_blocks)
descriptor_text = Text()
args.weights = [float(weight) for weight in args.weights]
descriptor_choice = {
    "color": [descriptor_color],
    "texture": [descriptor_texture],
    "text": [descriptor_text],
    "mix_color_text": [descriptor_color, descriptor_text],
    "mix_color_texture": [descriptor_color, descriptor_texture],
    "mix_text_texture": [descriptor_text, descriptor_texture],
    "mix_text_texture_color": [descriptor_text, descriptor_texture, descriptor_color],

}
if len(args.weights) != len(descriptor_choice[args.descriptor]):
    print('number of weights and descriptors do not match')
    exit()
elif sum(args.weights) != 1:
    print('weights must add up to 1')
    exit()
else:
    weights = args.weights
    print('Vector weights {} for descriptor {}'.format(weights,args.descriptor))
museum_similarity_comparator = museum.Museum(
    args.museum_images_path, descriptor_choice[args.descriptor], similarity_mode=args.similarity, rm_frame=True, rm_noise=args.rm_noise,
)

canvas = Canvas()
descriptor_text.load_bg_dataset_txt(museum_similarity_comparator.image_dataset)
query_image_path = args.query_image_path
if args.rm_background:
    CANVAS_TMP_FOLDER = os.path.join(query_image_path, CANVAS_TMP_FOLDER)
    CANVAS_TMP_FOLDER_CROPPED = os.path.join(
        query_image_path, CANVAS_TMP_FOLDER_CROPPED)
    if os.path.isdir(CANVAS_TMP_FOLDER):
        shutil.rmtree(CANVAS_TMP_FOLDER)
        shutil.rmtree(CANVAS_TMP_FOLDER_CROPPED)
    os.mkdir(CANVAS_TMP_FOLDER)
    os.mkdir(CANVAS_TMP_FOLDER_CROPPED)
    list_of_coords = []
    for image in sorted(os.listdir(query_image_path)):
        is_img = museum.Museum.file_is_image(
            os.path.join(query_image_path, image))
        if is_img:
            _, x, y, w, h, x2, y2, w2, h2 = canvas.background_remover(os.path.join(query_image_path, image), os.path.join(
                os.getcwd(), CANVAS_TMP_FOLDER), os.path.join(os.getcwd(), CANVAS_TMP_FOLDER_CROPPED), image)
            temp_list = [(x, y)]
            if x2 > 0:
                temp_list.append((x2, y2))
            list_of_coords.append(temp_list)
    results.create_results(list_of_coords, file_path=os.path.join(
        query_image_path, "coordinates_mask_original_image.pkl"))
    #query_image_path = CANVAS_TMP_FOLDER_CROPPED

final_result = []
if os.path.isdir(query_image_path):
    for original_image in sorted(os.listdir(query_image_path)):
        if museum_similarity_comparator.file_is_image(os.path.join(query_image_path, original_image)):
            image_set = ["crop_{}.jpg".format(original_image.split(".")[0]), "crop_{}_2.jpg".format(
                original_image.split(".")[0])] if args.rm_background else [original_image]
            img_path = CANVAS_TMP_FOLDER_CROPPED if args.rm_background else query_image_path
            img_set_res = []
            for image in image_set:
                if os.path.isfile(os.path.join(img_path, image)):
                    try:
                        # working multiscale
                        result = museum_similarity_comparator.compute_similarity(
                            os.path.join(img_path, image), text_extractor_method=descriptor_text.text_extraction
                        )
                        #print(result)
                        # working at given image size
                        #result = museum_similarity_comparator.compute_similarity(os.path.join(query_image_path, image), args.metric)
                    except museum.FileIsNotImageError:
                        continue
                    if len(weights) > 1:
                        result = weight_results_and_normalize_metrics(result, weights)
                    else:
                        result = result[0]    
                    result.sort(key=lambda x: x[1])  # resulting score sorted
                    result = result[:k]  # take the k elements
                    # For eache element, get only the image and forget about the actual similarity value
                    result = [key for key, val in result]
                    img_set_res.append(result)
            final_result.append(img_set_res)
            print(final_result)
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
        query_image_path, text_extractor_method=descriptor_text.text_extraction
    )
    # working at given image size
    #result = museum_similarity_comparator.compute_similarity(os.path.join(query_image_path, image), args.metric)
    result.sort(key=lambda x: x[1])  # resulting score sorted
    result = result[:k]
    # For eache element, get only the image and forget about the actual similarity value
    final_result = [key for key, val in result]

print("Final Guess")
print(final_result)
results.create_results(final_result)

# For evaluation purposes

# python3 main.py "datasets/BBDD" "datasets/qst1_w1/" "L1_norm" -g "datasets/qst1_w1/gt_corresps.pkl" --metric rgb_3d -k 1

# python3 main.py "datasets/BBDD" "datasets/qst2_w1/" "L1_norm" -g "datasets/qst2_w1/gt_corresps.pkl" --metric rgb_3d -k 1 --remove_back
