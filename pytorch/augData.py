# Data augmentation file
# Requires all base images to be in one folder, with all labeled images having the same name as a png in another folder
# Loops through all base images and applies a random number of transforms on the image, then applies the same to the labeled images

import numpy as np
import os
import random
from scipy import ndarray

import skimage as sk
from skimage import transform
from skimage import util
from skimage import io

# Set of transforms
def random_rotation(image_array):
    random_degree = random.uniform(-25, 25)
    return (sk.transform.rotate(image_array, random_degree), random_degree)

def set_rotation(image_array, degree):
    return sk.transform.rotate(image_array, degree)

def random_noise(image_array):
    return sk.util.random_noise(image_array)

def gaussian(image_array):
    return sk.filter.gaussian_filter(image_array, sigma = (3.0, 3.0))#, truncate = 3.5, multichannel = True)

def horizontal_flip(image_array):
    return image_array[:,::-1]

def vertical_flip(image_array):
    return image_array[::-1,:]

def random_shift(image_array):
    random_unit_x = random.uniform(-25, 25)
    random_unit_y = random.uniform(-25, 25)
    tform = sk.transform.SimilarityTransform(translation=(random_unit_x, random_unit_y))
    return (sk.transform.warp(image_array, tform), random_unit_x, random_unit_y)

def set_shift(image_array, shift_x, shift_y):
    tform = sk.transform.SimilarityTransform(translation=(shift_x, shift_y))
    return sk.transform.warp(image_array, tform)

def run():
    folder_path = '../data/all_navcam'
    train_folder_path = '../data/all_navcam/left_jpgs'
    label_folder_path = '../data/all_navcam/labels_pngs'

    train_images = [os.path.join(train_folder_path, f) for f in os.listdir(train_folder_path) if os.path.isfile(os.path.join(train_folder_path, f))]

    #label_images = [os.path.join(label_folder_path, f) for f in os.listdir(label_folder_path) if os.path.isfile(os.path.join(label_folder_path, f))]

    # Number of files that have already been generated and the total number of augmented files you desire
    num_generated_files = 5530
    num_files_desired = 50000
    total_files = len(train_images)
    #assert(len(train_images) == len(label_images))

    # Places available augmentations in a dictionary
    available_tfs = {
        'rotate': random_rotation,
        'noise': random_noise,
        'horizontal_flip': horizontal_flip,
        'vertical_flip': vertical_flip,
        'gaussian': gaussian,
        'shift': random_shift
        }

    # Main generation loop
    while num_generated_files <= num_files_desired:
        #train_to_transform, ind = next(iter(loaders['train']))
        #label_to_transform = datasets['train_label'][ind][0]
        
        # Randomly selects a base training image then pulls the corresponding labeled image using the name
        image_ind = random.randint(0, total_files-1)
        train_path = train_images[image_ind]
        #label_path = label_images[image_ind]
        label_path = label_folder_path+train_path[-29:-4]+".png"

        train_to_transform = sk.io.imread(train_path)
        label_to_transform = sk.io.imread(label_path)

        # Modify this to adjust the minimum and maximum number of transforms you wish to apply
        num_tfs_to_apply = random.randint(1, 5)

        num_tfs = 0
        tf_train = None
        tf_label = None
        while num_tfs <= num_tfs_to_apply:
            key = random.choice(list(available_tfs))
            if key == 'rotate':
                (tf_train, deg) = available_tfs[key](train_to_transform)
            elif key == 'shift':
                (tf_train, shift_x, shift_y) = available_tfs[key](train_to_transform)
            else:
                tf_train = available_tfs[key](train_to_transform)
                
            # Apply only affine transforms to the corresponding labeled image
            if key != 'noise' and key != 'gaussian':
                if key == 'rotate':
                    tf_label = set_rotation(label_to_transform, deg)
                elif key == 'shift':
                    tf_label = set_shift(label_to_transform, shift_x, shift_y)
                else:
                    tf_label = available_tfs[key](label_to_transform)
            print(key)
            num_tfs += 1

        new_train_path = '%s/aug_train/augment_train_%s.png' % (folder_path, num_generated_files)
        new_label_path = '%s/aug_label/augment_label_%s.png' % (folder_path, num_generated_files)

        # Save the generated images
        if type(tf_train) == np.ndarray and type(tf_label) == np.ndarray:
            io.imsave(new_train_path, tf_train)
            io.imsave(new_label_path, tf_label)
            print("Generated img_%d" % num_generated_files)
            num_generated_files += 1

    print("Finished Generating")
    return

if __name__ == "__main__":
    run()


