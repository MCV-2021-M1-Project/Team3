import argparse


import museum
from similarity import compute_similarity


import argparse

parser_epilog = \
"""
      Example given an image:\n

      python3 main.py "datasets/museum_set" "datasets/query_set/00002.jpg" "L1_norm"\n
      
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

parser.add_argument('--remove_back',
                      dest='rm_background', 
                      action='store_true',
                      help='Remove the paint background'
                      )
parser.set_defaults(rm_background=False)

# Parse arguments
args = parser.parse_args()


museum_similarity_comparator = museum.Museum(args.museum_images_path, rm_frame=False, similarity_mode=args.similarity, color_space=args.color_space)
result = museum_similarity_comparator.compute_similarity(args.query_image_path)
print(result)

