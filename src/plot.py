import numpy as np
import os
import matplotlib
# need to run in server
matplotlib.use('agg') 
from matplotlib import pyplot as plt
from metric import get_metrics, cvs_to_metric
from assistant import cvs_to_arr, avg_arr


def cvs_path_list(folder_to_read, requirement):
    """
    From folder to read, get the list of wanted .cvs path as a list

    folder_to_read: the path for folder
    requirement: 'logit' or 'loss' or 'predictions' cvs files

    return -> lists of cvs files [0] = train, [1] = test, [2] = validation
    """
    sets = ['train_data', 'test_data', 'val_data']
    cvs_list = []
    for data_set in sets:
        set_list = []
        files = os.listdir(folder_to_read + data_set)
        for order in range(500):
            for file in files:
                if file == 'ep_' + str(order) + '_' + requirement + '.csv':
                    set_list.append(folder_to_read + data_set + '/' + file)
        cvs_list.append(set_list)

    return cvs_list


def read_loss_data(folder_to_read):
    """
    Read folder and generate numpy array with average loss for every epoch

    folder_to_read: path of folder

    return -> numpy array; all_arr_loss[0] = loss of train_data, [1] = loss of test_data, [2] = loss of val_data
    """
    loss_cvs_files = cvs_path_list(folder_to_read, 'loss')
    all_arr_loss = []
    # loss_cvs_files[0] = 'train_data', [1] = 'test_data', [2] = 'val_data'
    for i in range(len(loss_cvs_files)):
        avg_loss = []
        for loss_path in loss_cvs_files[i]:
            arr = cvs_to_arr(loss_path)
            # print('arr', len(arr), file, data_set)
            avg_loss.append(avg_arr(arr))
        all_arr_loss.append(avg_loss)
    return all_arr_loss


def read_prediction_data(folder_to_read, metric='accuracy'):
    """
    Read folder and generate metric array

    folder_to_read: path of folder
    metric: 'accuracy', 'sensitivity', 'specificity'

    return -> desired_metric[0] = train_metric, [1] = test_metric, [2] = val_metric
    """
    cvs_files = cvs_path_list(folder_to_read, 'predictions')
    desired_metric = []
    # cvs_file[0] = 'train_data', [1] = 'test_data', [2] = 'val_data'
    for i in range(len(cvs_files)):
        metric_list = []
        for file in cvs_files[i]:
            arr = cvs_to_arr(file)
            metric_value = get_metrics(arr)
            if metric == 'Accuracy':
                metric_list.append(metric_value[0])
            if metric == 'Sensitivity':
                metric_list.append(metric_value[1])
            if metric == 'Specificity':
                metric_list.append(metric_value[2])
        desired_metric.append(metric_list)
    return desired_metric


def arr_to_plot(y_axis_arr, y_axis_name, legend, line_color, line_width=1):
    """
    Plotting with listed values along to epoch

    y_axis_values: numbers as list
    y_axis_name: loss or accuracy or sensitivity or specificity
    legend: wanted legend
    lw: line_width

    return -> plot of loss/accuracy/prediction/sensitivity/specificity vs epoch
    """

    plt.xlim(0, len(y_axis_arr))
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Epoch')
    plt.ylabel(y_axis_name)
    plt.plot(y_axis_arr, label=legend, color=line_color, lw=line_width)


if __name__ == "__main__":
    # folder_to_read = '../results_monday/no_aug_yes_pretrain_results/'
    folder_to_read = '../large_images_aug_yes_pretrain_results_v2/'


    # ---------------------- plot loss ---------------------------------------
    arr_to_plot(read_loss_data(folder_to_read)[0], 'Loss', legend='Train loss', line_color='black', line_width=3)
    arr_to_plot(read_loss_data(folder_to_read)[2], 'Loss', legend='Validation loss', line_color='blue')
    arr_to_plot(read_loss_data(folder_to_read)[1], 'Loss', legend='Test loss', line_color='red')
    plt.legend(loc='upper right', fontsize=20)
    # to remove first and last tick of y-axis
    ax = plt.gca()
    ax.set_yticks(ax.get_yticks()[1:-1])
    plt.savefig(folder_to_read + 'Loss_plot.png')
    plt.show()
    plt.close()

    # ---------------------- plot loss ---------------------------------------
    metrics_types = ['Accuracy', 'Sensitivity', 'Specificity']
    for item in metrics_types:
        arr_to_plot(read_prediction_data(folder_to_read, metric=item)[0], item, legend='Train ' + item, line_color='black', line_width=3)
        arr_to_plot(read_prediction_data(folder_to_read, metric=item)[2], item, legend='Validation ' + item, line_color='blue')
        arr_to_plot(read_prediction_data(folder_to_read, metric=item)[1], item, legend='Test ' + item, line_color='red')
        plt.legend(loc='lower right', fontsize=20)
        # to remove first and last tick of y-ax
        ax = plt.gca()                       
        ax.set_yticks(ax.get_yticks()[1:-1])
        plt.savefig(folder_to_read + '{}_plot.png'.format(item))
        plt.show()
        plt.close()



