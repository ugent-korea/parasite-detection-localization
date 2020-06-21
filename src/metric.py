import numpy as np
import pandas as pd
from assistant import cvs_to_arr


def get_metrics(pred_arr):
    """
    Get metrics from predictions

    arr : prediction_array

    return -> metric values; accuracy[0], sensitivity[1], specificity[2]
    """
    arr = pred_arr.transpose(1, 0)
    pred = arr[1]                               # prediction value
    corr_label = arr[0]                         # true value
    tp_tn = pred == corr_label                  # if pred == corr_label, it returns True else False
    correct_pred = tp_tn.astype(int)
    tp, tn, fp, fn = 0, 0, 0, 0
    # count = 0
    for index in range(0, len(arr[0])):
        if correct_pred[index] == 1:
            # count += 1
            if corr_label[index] == 1:
                tp += 1
            elif corr_label[index] == 0:
                tn += 1
        elif correct_pred[index] == 0:
            # count += 1
            if corr_label[index] == 0:
                fp += 1
            elif corr_label[index] == 1:
                fn += 1

    # print('tp',tp)
    # print('np',tn)
    # print('fp',fp)
    # print('fn',fn)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)                # recall = sensitivity
    specificity = tn / (tn + fp)

    # print("accuracy: %.3f" % accuracy +"\n" + "sensitivity: %.3f" % sensitivity + "\n" + "specificity: %.3f" %
    # specificity)
    return accuracy, sensitivity, specificity


def cvs_to_metric(cvs_path, requirement):
    """
    Read prediction.csv files and make metrics

    requirement: accuracy or sensitivity or specificity

    return -> wanted metric array
    """
    requirements = []
    for item in cvs_path:
        arr = cvs_to_arr(item)
        metric = get_metrics(arr)
        if requirement == 'Accuracy':
            requirements.append(metric[0])
        if requirement == 'Sensitivity':
            requirements.append(metric[1])
        if requirement == 'Specificity':
            requirements.append(metric[2])
    return requirements


if __name__ == "__main__":
    arr = cvs_to_arr(cvs_file='/example_results/train_data/ep_0_predictions.csv')
    metrics = get_metrics(arr)
    print(metrics[0])

