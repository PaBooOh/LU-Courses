import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

def rescale_images(scaled_size=(64,64), dataset_folder_path = 'images/original_img/', dataset_scaled_folder_path = 'images/scaled_img/'):
    data_fileList = os.listdir(dataset_folder_path)
    for data_file in data_fileList:
        in_img = Image.open(dataset_folder_path + data_file)
        # (x, y) = in_img.size
        out_img = in_img.resize(scaled_size, Image.ANTIALIAS)
        out_img.save(dataset_scaled_folder_path + data_file)

def images_to_RGB(dataset_scaled_folder_path = 'images/scaled_img/', dataset_text_path = 'images/anime.npy'):
    data_fileList = os.listdir(dataset_scaled_folder_path)
    dataset_text = []
    for data_file in data_fileList:
        # img = misc.imread(dataset_scaled_folder_path + data_file)
        img = Image.open(dataset_scaled_folder_path + data_file)
        x, y = img.size
        rgb = np.array(img.getdata()).reshape((x, y, -1))
        dataset_text.append(rgb)
    
    dataset_text = np.array(dataset_text)
    np.save(dataset_text_path, dataset_text)
        # print(rgb)
        # print(rgb.shape)
        # plt.imshow(rgb)
        # plt.show()
        # break

# X = np.load('images/anime.npy')
# plt.imshow(X[2467])
# plt.show()
# print()