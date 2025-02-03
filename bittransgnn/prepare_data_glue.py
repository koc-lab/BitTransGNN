from datasets import load_dataset
import numpy as np

# preparation for datasets listed under GLUE benchmark

if __name__ == "__main__":
    datasets = ["cola", "mrpc", "rte", "stsb", "wnli"]

    for dataset_name in datasets:
        f = open('dataset/corpus/' + dataset_name + '.clean.txt', 'r')

        min_len = 10000
        aver_len = 0
        max_len = 0 

        #f = open('data/wiki_long_abstracts_en_text.txt', 'r')
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            temp = line.split()
            aver_len = aver_len + len(temp)
            if len(temp) < min_len:
                min_len = len(temp)
            if len(temp) > max_len:
                max_len = len(temp)
        f.close()
        aver_len = 1.0 * aver_len / len(lines)
        print('min_len : ' + str(min_len))
        print('max_len : ' + str(max_len))
        print('average_len : ' + str(aver_len))
        print(f"txt docs prepared for {dataset_name} dataset.")
