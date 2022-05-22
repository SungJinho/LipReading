import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd
import json


if __name__ == '__main__':

    # Folder path
    json_folder_path = '../labeling'
    image_folder_path = '../image'
    json_folder_list = os.listdir(json_folder_path)

    # Make file list
    json_list = []
    
    for currentdir, dirs, files in os.walk(json_folder_path):
        for file in files :
            json_list.append(currentdir + '/' + file)            
    print('json file num: ', len(json_list))  
    
    # Split train, test data
    file_idx = np.array(range(0, len(json_list)))
    train_idx, test_idx = train_test_split(file_idx, random_state=66, test_size=0.2)
    print('[train, test]: ', len(train_idx), len(test_idx))

    # Make test CSV
    test_data = []
    header = ['keypoint']
    for i in range(0, 48):
        header.append(i)
    test_data.append(header)
    for i in test_idx:
        #print(file_list[i])
        
        with open(json_list[i]) as jf:
            json_data = json.load(jf)
            annotations = json_data['annotations']
            keypoint = annotations[0]['2D keypoints']
            image_name = annotations[0]['image_name']

            a = json_list[i].split('/')
            image_path = image_folder_path + '/' + a[2] + '/' + a[3] + '/' + image_name

            key_list = []
            key_list.append(image_path)

            for n in range(0, 24):
                key_list.append(keypoint[3*n + 0])
                key_list.append(keypoint[3*n + 1])
            
            test_data.append(key_list)

    df_test = pd.DataFrame(test_data)
    df_test.to_csv('test.csv', index=False, header=False)

    # Make train CSV
    train_data = []
    train_data.append(header)

    for i in train_idx:
        
        with open(json_list[i]) as jf:
            json_data = json.load(jf)
            annotations = json_data['annotations']
            keypoint = annotations[0]['2D keypoints']
            image_name = annotations[0]['image_name']

            a = json_list[i].split('/')
            image_path = image_folder_path + '/' + a[2] + '/' + a[3] + '/' + image_name

            key_list = []
            key_list.append(image_path)

            for n in range(0, 24):
                key_list.append(keypoint[3*n + 0])
                key_list.append(keypoint[3*n + 1])
            
            train_data.append(key_list)

    df_train = pd.DataFrame(train_data)
    df_train.to_csv('train.csv', index=False, header=False)








  