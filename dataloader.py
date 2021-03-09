import os
import shutil 
import time
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torchvision.datasets import MNIST, CelebA
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import numpy as np 
from skimage import io, transform

from models import VAE


def mnist_check():
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Chrome')]
    urllib.request.install_opener(opener)



def image_selection():
    output_path = "data/celeba/img_align_celeba/selected" #"Users⁩/gchou⁩/⁨Downloads⁩/⁨archive⁩/⁨img_align_celeba⁩/selected"
    image_path = "data/celeba/img_align_celeba/img_align_celeba"
    celeba_attr_file = "data/celeba/list_attr_celeba_original.txt"
    attr_types = [5, 6, 9, 10, 11, 16, 21, 23, 32, 36] 
    # bald, bangs, black hair, blond hair, blurry, eyeglasses, male, mustache, smiling, wearing hat 
    #attr_type=16

    count = 0
    count_total = 0

    with open(celeba_attr_file, "r") as attr_file:
        count_total += 1
        attr_info = attr_file.readlines()
        attr_info = attr_info[1:]
        for line in attr_info:
            info = line.split()
            filename = info[0]
            filepath_old = os.path.join(image_path, filename)

            if os.path.isfile(filepath_old):

                for attr_type in attr_types:
                    if int(info[attr_type]) == 1:
                        filepath_new = os.path.join(output_path, filename)
                        shutil.copyfile(filepath_old, filepath_new)
                        count += 1
                        continue 

    print("There were {} selected images out of {} total images".format(count, count_total))


def attr_annotation_selection():
    attr_types = [5, 6, 9, 10, 11, 16, 21, 23, 32, 36] 
    # bald, bangs, black hair, blond hair, blurry, eyeglasses, male, mustache, smiling, wearing hat 

    original_attr_file = "data/celeba/list_attr_celeba_original.txt"
    new_attr_file = open("data/celeba/list_attr_celeba.txt","w")

    count = 0
    idx = 0
    with open(original_attr_file, "r") as attr_file:
        attr_info = attr_file.readlines() #reads entire file 
        attr_info = attr_info[2:] #skips first two lines, which are total count and feature names, respectively

        for line in attr_info:
            info = line.split()
            new_feature_array = [info[0]]
            for idx, attr_type in enumerate(info[1:], start=1):
                feature = 1 if int(attr_type) == 1 else 0 # 0 and 1 instead of -1 and 1 
                if idx in attr_types:
                    new_feature_array.append(feature)

            if not np.all((np.array(new_feature_array[1:])==0)):
                print((new_feature_array))
                new_attr_file.write("{} ".format(str(new_feature_array[0])))
                new_attr_file.write("{}\n".format(" ".join(str(e) for e in new_feature_array[1:])))
                count += 1
                 
        idx += 1

    new_attr_file.close()
    print("{} lines were copied".format(count))

def get_celeba_selected_dataset(attr_file="data/celeba/list_attr_celeba.txt", im_dir="data/celeba/img_align_celeba/img_align_celeba"):

    def display_samples():
        def show_annotations(image, annotations):
            """Show image with landmarks"""
            plt.imshow(image)
            plt.text(0, 0, "c={}".format(annotations), color='black',
                                    backgroundcolor='white', fontsize=8)
        fig = plt.figure()
        for i in range(len(face_dataset)):
            sample = face_dataset[i]
            print(i, sample['image'].shape, sample['annotations'].shape)
            ax = plt.subplot(1, 4, i + 1)
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(i))
            ax.axis('off')
            show_annotations(**sample)
            if i == 3:
                plt.show()
                break


    class SelectedCelebaDataset(Dataset):
        '''Celeba dataset with selected features (e.g. 10 out of 40)'''

        def __init__(self, attr_file, im_dir, transform=None):
            """
            Args:
                attr_file (string): Path to the attr txt file with annotations.
                im_dir (string): Directory with all the images (selected).
                transform (callable, optional): Optional transform to be applied
                    on a sample.
            """
            self.attr_file = attr_file
            with open(attr_file, "r") as attrs:
                self.attr_info = attrs.readlines() # list of all the annotations 
            self.im_dir = im_dir
            self.transform = transform

        def __len__(self):
            return len(self.attr_info)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            img_name = os.path.join(self.im_dir, self.attr_info[idx].split()[0])
            image = io.imread(img_name)
            annotations = self.attr_info[idx].split()[1:] #list of strings ['0','1',...]
            annotations = np.array([int(i) for i in annotations])
            #annotations = annotations.astype('float')#.reshape(-1, 1)

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W

            image = torch.from_numpy(image)
            image = torch.transpose(image,0,2)
            image = torch.transpose(image,1,2)
            # the three lines above should be equivalent to first transposing the nparray: image = image.transpose((2, 0, 1))
            # then converting to torch and then dividing by 255.0

            image = image/255.0

            # if self.transform:
            #     image = self.transform(torch.from_numpy(image).float())
            # else: 
            #     image = torch.from_numpy(image) / 255.0 #normalize 
   
            return image, torch.from_numpy(annotations).float()


    celeba_selected_dataset = SelectedCelebaDataset(attr_file, im_dir)#, transform=transforms.Compose([transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])]))

    return celeba_selected_dataset


if __name__ == '__main__':
    #attr_annotation_selection()
    #image_selection()
