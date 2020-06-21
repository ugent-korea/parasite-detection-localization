"""
Created on Wed Apr  8 13:39:32 2020

@author: yeji_bae
"""
import glob
import random
import numpy as np
import torch
from PIL import Image
# import torch.nn as nn
# import torchvision.models as models
from torch.utils.data.dataset import Dataset
from Image_operation import crop_image, crop_center_point, rotate_image, flip_image, change_brightness, \
    uniform_noise_image, array_to_image


class ParasiteDatasetTrain(Dataset):

    def __init__(self, folder_to_read):
        self.all_image_list = glob.glob(folder_to_read + '*.png')
        self.image_amt = len(self.all_image_list)
        self.im_class_flag = 0

    def __getitem__(self, index):
        # Read image location
        while True:
            # print('LOOPING')
            im_index = random.randint(0, len(self.all_image_list) - 1)
            image_path = self.all_image_list[im_index]

            # Determine class 0:negative data , 1:positive data
            if 'negative' in image_path:
                im_class = 0
            elif 'vid' in image_path:
                im_class = 1

                # To balance the positive and negative dataset; (if starts with 0, then 1,0,1,0,1)
            if self.im_class_flag == im_class:
                other_class = [0, 1]
                other_class.remove(im_class)
                self.im_class_flag = other_class[0]
                break

        # Read image
        im = Image.open(image_path)
        # Convert to array
        im_arr = np.array(im)

        # --- IMAGE OPERATIONS BEGIN ---
        # Crop_randomly
        height, width = im_arr.shape[0], im_arr.shape[1]
        h, w = int(np.random.choice(height - 224, 1)), int(np.random.choice(width - 224, 1))
        im_arr = crop_image(im_arr, (h, w), (224, 224))
        # Rotate_randomly
        degree = random.choice([0, 1, 2, 3])
        im_arr = rotate_image(im_arr, degree)
        # Flip left and right
        choice = random.choice([0, 1])
        if choice == 0:
            im_arr = flip_image(im_arr, 1)
        else:
            pass
        # Brightness
        if random.choice([0, 1]):
            choice = random.randrange(-5, 5, 1)
            im_arr = change_brightness(im_arr, choice)
        # Uniform noise
        if random.choice([0, 1]):
            im_arr = uniform_noise_image(im_arr, 0, 5)
        # Just in case if the image has values less than 0 or more than 255 with
        im_arr[im_arr > 255] = 255
        im_arr[im_arr < 0] = 0
        # Invert colors
        if False: # random.choice([0, 1]):
             im_arr = 255 - im_arr
        # --- IMAGE OPERATIONS END ---
        # Normalize
        im_arr = im_arr / 255
        # Gray scale(one channel) to RGB scale(three channel)
        im_arr = np.stack([im_arr, im_arr, im_arr], axis=0)

        im_tensor = torch.from_numpy(im_arr).float()
        return im_tensor, im_class

    def __len__(self):
        return 2000


class ParasiteDatasetTest(Dataset):

    def __init__(self, folder_to_read):
        self.all_image_list = glob.glob(folder_to_read + '*.png')
        self.image_amt = len(self.all_image_list)

    def __getitem__(self, index):
        # Read image location
        image_path = self.all_image_list[index]

        # Determine class
        if 'negative' in image_path:
            im_class = 0
        elif 'vid' in image_path:
            im_class = 1

        # Read image
        im = Image.open(image_path)
        # Convert to array
        im_arr = np.array(im)

        # --- IMAGE OPERATIONS BEGIN ---
        # center crop
        center_crop_point = crop_center_point(im_arr, (224, 224))
        im_arr = crop_image(im_arr, center_crop_point, (224, 224))
        # Normalize
        im_arr = im_arr / 255
        # --- IMAGE OPERATIONS END ---

        # Gray scale to RGB scale
        im_arr = np.stack([im_arr, im_arr, im_arr], axis=0)

        im_tensor = torch.from_numpy(im_arr).float()
        return im_tensor, im_class

    def __len__(self):
        return self.image_amt


if __name__ == "__main__":
    # input data path
    para_dataset_tr = ParasiteDatasetTrain(folder_to_read='../dataset/train_frames/')
    for x in range(100):
        a, b = para_dataset_tr[0]
        a = a * 255
        a = a.numpy().transpose(1, 2, 0)
        a = np.uint8(a)
        a = Image.fromarray(a)
        a.save(str(x)+'_.png')


    # print(len(para_dataset_tr))
    # print(a, b)
    # print(torch.max(a), torch.min(a))

    """
    para_dataset_ts = ParasiteDatasetTest(folder_to_read='../dataset/test_frames/')
    print(len(para_dataset_ts))
    a, b = para_dataset_ts[0]
    # print(a, b)
    # print(torch.max(a), torch.min(a))
    """

    """

    # # show in image
    # b = b.numpy() * 255
    # back_to_im = Image.fromarray(b)
    # back_to_im.show()

    para_dataset_loader = torch.utils.data.DataLoader(dataset=para_dataset_tr,
                                                      batch_size=20,
                                                      shuffle=False)
    model = models.vgg11()
    model.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)
    # print(model)
    # #  -> print architecture
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #print(len(para_dataset_loader))

    for index, (images, labels) in enumerate(para_dataset_loader):
        # print(images.shape)   # (224,224) , 1,2,...   what is this (224,224)repetition
        # print(images)         # (224,224) , tensor
        # print(labels)         # (224,224) class
        # Clear gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        # Calculate loss
        loss = criterion(outputs, labels)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        # print('output', outputs)
        # print(outputs.size())
        break
    """
