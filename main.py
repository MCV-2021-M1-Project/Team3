import argparse
import os
import argparse


import museum
from similarity import compute_similarity
import results
from mapk import mapk

parser_epilog = \
"""
      Example given an image:\n

      python3 main.py "datasets/museum_set" "datasets/qsd1_w1/00002.jpg" "L1_norm"\n
      
      Example given a quey path:\n
      
      python3 main.py "datasets/museum_set" "datasets/query_set/" "L1_norm"
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

parser.add_argument('-c', '--color_space',
                       metavar='color_space',
                       type=str,
                       choices=["gray", "rgb", "hsv", "lab"],
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
gt = results.ground_truth("datasets/qsd1_w1/gt_corresps.pkl")

museum_similarity_comparator = museum.Museum(args.museum_images_path, rm_frame=True, similarity_mode=args.similarity, color_space=args.color_space)
final_result = []
if os.path.isdir(args.query_image_path):
    for image in sorted(os.listdir(args.query_image_path)):
        try:
            result = museum_similarity_comparator.compute_similarity(os.path.join(args.query_image_path, image))
        except museum.FileIsNotImageError:
            continue
        result.sort(key=lambda x: x[1], reverse=True) # resulting score sorted
        result = result[:k] # take the k elements
        result = [ key for key, val in result] ## For eache element, get only the image and forget about the actual similarity value
        final_result.append(result)
        print(final_result)
    mapk_result = mapk(gt, final_result, k=k)
    print("Resulting Mapk:")
    print(mapk_result)
else:
    result = museum_similarity_comparator.compute_similarity(args.query_image_path)
    result.sort(key=lambda x: x[1], reverse=True) # resulting score sorted
    result = result[:k]
    final_result = [ key for key, val in result] ## For eache element, get only the image and forget about the actual similarity value


# For evaluation purposes
"""
res = []
gt = results.ground_truth("datasets/qsd1_w1/gt_corresps.pkl")
for indx, query in enumerate(final_result):
    print("query: " + str(query))
    print("gt: " + str(gt[indx]))
    res.append(query == gt[indx])
print(sum(res))
"""

# Results for each similarity measure
#cosine : 7
#L1 : 2
#L2 : 1
#histogram_intersection: 2
#hellinger_similarity: 0
