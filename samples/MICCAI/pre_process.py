"""

From the basis miccai's dataset, this scipt remove all images with consensus mask with less than treshold_pixel pixels.

User must specify original dataset and destination dataset. However, these 2 folders must exists. So user must create them.
Destination dataset must be a folder with sub-folders for every patients.
This script will copy image's slices and their corresponding consensus image.

"""
import os
import sys
#import datetime
import numpy as np
#import skimage.draw
import SimpleITK as sitk
import matplotlib.pylab as plt
from skimage import measure
import shutil

def separate_mask(mask):
    all_mask = []
    Is_mask = False
    treshold_pixel = 20
    # will contain all mask generated from mask
    nb_label = 0
    label_mask, nb_label = measure.label(mask, connectivity=1, return_num=True)
    # connectivity=1 for 4 neighbors, 2 for 8 neighbors
    if nb_label >= 1:
        for i in range(1, nb_label + 1):
            mask_compare = np.full(np.shape(label_mask), i)
            # check equality test and have the value i on the location of each mask
            separate_mask = np.equal(label_mask, mask_compare).astype(int)
            # give new name to the masks
            count_non_zero = np.count_nonzero(separate_mask)
            #print(" count non zeros : ", count_non_zero)
            if count_non_zero > treshold_pixel: 
                # take this mask
                all_mask.append(separate_mask)

        if all_mask == []:
            # no mask found
            mask = []
        else:
            mask = np.stack(all_mask, axis=-1)
            Is_mask = True
    else:
        # no mask found
        mask = []

    return mask, Is_mask

############################################################
#  Dataset
############################################################
def load_dataset(dataset_dir):

    image_ids = [] # contains complete path of images

    #print("dataset_dir = ", dataset_dir)

    # r=root, d=directories, f = files
    for r, d, f in sorted(os.walk(
            dataset_dir)):  # !MB to choose the same patients always for train and val, useful for transfer learning
        for file in f:
            if '3DFLAIR' in file:
                #if (int(file[8:11]) >= coupe_min and int(file[8:11]) < coupe_max):
                image_ids.append(os.path.join(r, file))

    #for image_id in image_ids[256:260]:
        #print("image_id path = ", image_id)

    return image_ids

def load_mask(image_id, dataset_cleaned):

    img_name = os.path.basename(image_id)
    #print(" img_name = ", img_name)
    img_num = img_name[len("3DFLAIR_"):len(img_name)-len(".nii.gz")]
    #print(" img_num = ", img_num, len(img_num))
    mask_id = os.path.join(os.path.dirname(image_id), "Consensus_" + str(img_num) + ".nii.gz")
    mask_name = os.path.basename(mask_id)
    #print(" mask_id = ", mask_id)

    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_id, sitk.sitkUInt8))

    #print("mask shape = ", mask.shape)

    mask, Is_mask = separate_mask(mask)

    if Is_mask:
        # there is a mask
        #print("mask shape = ", mask.shape)
        #print("mask complete path = ", mask_id)
        #print("image name ", os.path.basename(image_id))
        patient_id = os.path.basename(image_id[:len(image_id)-len(img_name)-1])
        #print(" patient id 2 = ", patient_id)
        #print("new chemin ", patient_id+mask_name)
        #print("new chemin ", os.path.join(dataset_cleaned, os.path.join(patient_id, mask_name)))
        #print("chemin = ", os.path.join(dataset_cleaned, "dataset_cleaned"))
        #filePath = shutil.copy(mask_id, os.path.join(dataset_cleaned, os.path.join(patient_id, mask_name)))
        #filePath = shutil.copy(image_id, os.path.join(dataset_cleaned, os.path.join(patient_id, img_name)))



def load_image(image_id):

    #print("image complete path = ", image_id)

    image = sitk.GetArrayFromImage(sitk.ReadImage(image_id, sitk.sitkFloat32))
    #print("image shape = ", image.shape)

    #image[image < 0] = 0

    #print("image shape = ", image.shape)

    #l,c = np.where(image<0)
    #print(len(l))
    # print("shape image : ", image.shape)
    return image

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
#print("root dir = ", ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# original dataset
dataset_dir = os.path.join(ROOT_DIR, "dataset_miccai")
# destination dataset
dataset_cleaned = os.path.join(ROOT_DIR, "dataset_cleaned_20px")

print(" original dataset = ", dataset_dir)
print(" destination dataset = ", dataset_cleaned)
image_ids = load_dataset(dataset_dir)

#image = load_image(image_ids[12])
#load_mask(image_ids[128], dataset_cleaned)

# number of slices
nb_coupes = 256

cpt = 0
for image_id in image_ids:
    if cpt != nb_coupes:
        load_mask(image_id, dataset_cleaned)
    else:
        print("patient traite : ", os.path.basename(image_id[:len(image_id)-len(os.path.basename(image_id))-1]))
        cpt = 0
    cpt += 1
