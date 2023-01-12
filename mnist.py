# Code modified from: https://github.com/hsjeong5/MNIST-for-Numpy

import numpy as np
from urllib import request
import gzip
import pickle
import os
import pathlib
import cv2

# https://github.com/zalandoresearch/fashion-mnist
# https://github.com/rois-codh/kmnist
# https://github.com/darshanbagul/USPS_Digit_Classification/blob/master/USPSdata/USPSdata.zip
# http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip


def download_mnist():

    filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
    ]
    SAVE_PATH = pathlib.Path("data/original_mnist")

    SAVE_PATH.mkdir(exist_ok=True, parents=True)
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        filepath = SAVE_PATH.joinpath(name[1])
        if filepath.is_file():
            continue
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], filepath)


def extract_mnist(ds_mnist='mnist'):
    filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
    ]
    SAVE_PATH = pathlib.Path("data/original_" + ds_mnist)
    save_path = SAVE_PATH.joinpath(ds_mnist + ".pkl")
    if save_path.is_file():
        return
    mnist = {}
    # Load images
    for name in filename[:2]:
        path = SAVE_PATH.joinpath(name[1])
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            print(data.shape)
            mnist[name[0]] = data.reshape(-1,   28*28)
    # Load labels
    for name in filename[2:]:
        path = SAVE_PATH.joinpath(name[1])
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
            mnist[name[0]] = data
    with open(save_path, 'wb') as f:
        pickle.dump(mnist, f)
    
    
def load(ds_mnist='mnist'):
    if ds_mnist == 'mnist':
        download_mnist()
    
    if ds_mnist != 'usps':
        extract_mnist(ds_mnist)

        SAVE_PATH = pathlib.Path("data/original_" + ds_mnist)
        dataset_path = SAVE_PATH.joinpath(ds_mnist + ".pkl")
        with open(dataset_path, 'rb') as f:
            mnist = pickle.load(f)
        X_train, Y_train, X_test, Y_test = mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

    else:
        path_to_train_data = "data/original_usps/USPSdata/Numerals/"
        path_to_test_data = "data/original_usps/USPSdata/Test/"
        
        X_test = [] 
        Y_test = []
        img_list = os.listdir(path_to_test_data)
        for img_name in img_list:
            if '.png' in img_name:
                parts = img_name[:-4].split('_')
                i = int(parts[1])
                label_data = 9 - ( i // 150 )
                #print(img_name)
                #print(label_data)
            
                img = cv2.imread(path_to_test_data + img_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = 255 - cv2.resize(img, (28, 28))
                #print(img.min())
                #print(img.max())
                X_test.append(img)
                Y_test.append(label_data)
        X_test = np.array(X_test).astype(np.uint8)
        Y_test = np.array(Y_test)

        X_train = [] 
        Y_train = []
        for i in range(10):
            label_data = path_to_train_data + str(i) + '/'
            img_list = os.listdir(label_data)
            for name in img_list:
                if '.png' in name:
                    img = cv2.imread(label_data + name)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = 255 - cv2.resize(img, (28, 28))

                    X_train.append(img)
                    Y_train.append(i)
        X_train = np.array(X_train).astype(np.uint8)
        Y_train = np.array(Y_train)

    return X_train.reshape(-1, 28, 28), Y_train, X_test.reshape(-1, 28, 28), Y_test


if __name__ == '__main__':
    init()