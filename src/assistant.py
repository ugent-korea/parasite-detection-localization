import os
import torch
import numpy as np
import pandas as pd


def write_to_csv(file_name, arr):
    """
    Writing as csv form

    file_name: name of file to be created
    arr: input information as array

    return -> csv file will be formed
    """
    file_to_write = open(file_name, 'a')
    for item in arr:
        file_to_write.write(str(item) + ',')
    file_to_write.write('\n')
    file_to_write.close()
    return 1


def cvs_to_arr(cvs_file):  ## -> assitant
    """
    Read cvs file via pandas and change into numpy array

    return -> numpy array of cvs values
    """
    read_cvs = pd.read_csv(cvs_file, header=None).values  # read .cvs files via pandas df to array
    read_cvs = np.asarray(read_cvs)
    # print(read_cvs.ctypes)
    return read_cvs


def avg_arr(np_arr):
    """
    Get average of fist column of arr or list
    """
    sum = 0
    for item in np_arr:
        sum += item[0]
    return sum/len(np_arr)


def save_models(model, path, epoch):
    """
    Save models to given path

    model: model to be saved
    path: path that the model would be saved
    epoch: the epoch the model finished training
    """
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model, path + "/model_epoch_{0}.pt".format(epoch))
