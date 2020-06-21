import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data.dataset import Dataset
from learning import train_model, validate_model
from thesis_dataset import ParasiteDatasetTrain, ParasiteDatasetTest
from assistant import save_models
from metric import cvs_to_arr, get_metrics
# from torchsummary import summary

if __name__ == "__main__":
    para_tr_data = ParasiteDatasetTrain(folder_to_read='../no_resize_data/train_data/')
    tr_data_loader = torch.utils.data.DataLoader(dataset=para_tr_data,
                                                 batch_size=32,  # binary number is good for programming
                                                 shuffle=True,
                                                 num_workers=15)

    para_tr_noaug_data = ParasiteDatasetTest(folder_to_read='../no_resize_data/train_data/')
    tr_noaug_data_loader = torch.utils.data.DataLoader(dataset=para_tr_noaug_data,
                                                       batch_size=32,
                                                       shuffle=True,
                                                       num_workers=10)

    para_ts_noaug_data = ParasiteDatasetTest(folder_to_read='../no_resize_data/test_data/')
    ts_noaug_data_loader = torch.utils.data.DataLoader(dataset=para_ts_noaug_data,
                                                       batch_size=32,
                                                       shuffle=False,
                                                       num_workers=10)

    para_val_noaug_data = ParasiteDatasetTest(folder_to_read='../no_resize_data/val_data/')
    val_noaug_data_loader = torch.utils.data.DataLoader(dataset=para_val_noaug_data,
                                                        batch_size=32,
                                                        shuffle=False,
                                                        num_workers=10)


    DEVICE_ID = 0
    # set up for training
    folder = 'train_data'
    import os
    model_dir = "../large_images_aug_yes_pretrain_results_v2"
    os.makedirs(model_dir+'/train_data/')
    os.makedirs(model_dir+'/test_data/')
    os.makedirs(model_dir+'/val_data/')
    os.makedirs(model_dir+'/saved_models/')
    suffix = '/train'

    model = models.vgg16(pretrained=True)
    # modify the last Fc layer into size 2 for positive sample and negative sample
    model.classifier[6] = nn.Linear(in_features=4096, out_features=2,
                                    bias=True)  #### what does this "6" of classifier[6] ??????
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(100):
        # print('\nepoch %d:' % epoch)
        # train the model
        train_model(model, tr_data_loader, criterion, optimizer, gpu_id=DEVICE_ID)
        if epoch % 1 == 0:
            print('\nepoch %d:' % epoch)
            # validate the trained model for every 1 epoch
            val = validate_model(model, tr_noaug_data_loader, criterion, epoch,
                                 prediction_folder = model_dir+'/train_data/', gpu_id=DEVICE_ID)
            cvs_path = 'ep_' + str(epoch) + '_predictions.csv'
            arr = cvs_to_arr(cvs_file = model_dir+'/train_data/' + cvs_path)
            metrics = get_metrics(arr)
            print('train_accuracy', metrics[0])

            # validate the test model for every 1 epoch
            val = validate_model(model, ts_noaug_data_loader, criterion, epoch,
                                 prediction_folder = model_dir + '/test_data/', gpu_id=DEVICE_ID)
            cvs_path = 'ep_' + str(epoch) + '_predictions.csv'
            arr = cvs_to_arr(cvs_file = model_dir + '/test_data/' + cvs_path)
            metrics = get_metrics(arr)
            print('test_accuracy', metrics[0])

            # validate the validation model for every 1 epoch
            val = validate_model(model, val_noaug_data_loader, criterion, epoch,
                                 prediction_folder = model_dir + '/val_data/', gpu_id=DEVICE_ID)
            cvs_path = 'ep_' + str(epoch) + '_predictions.csv'
            arr = cvs_to_arr(cvs_file = model_dir + '/val_data/' + cvs_path)
            metrics = get_metrics(arr)
            print('val_accuracy', metrics[0])

        if epoch % 10 == 0:
            # save model
            save_models(model, model_dir+'/saved_models/', epoch)

        if epoch == 10:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        if epoch == 40:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
