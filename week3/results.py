import pickle

def ground_truth(results_file):
    f=open(results_file, 'rb')
    ground_truth_val = pickle.load(f)
    return ground_truth_val

def create_results(result_list, file_path="result.pkl"):
    with open(file_path, 'wb') as pkl_file:
        pickle.dump(result_list, pkl_file)

if __name__ == "__main__":
    print(ground_truth("datasets/qsd1_w1/gt_corresps.pkl"))