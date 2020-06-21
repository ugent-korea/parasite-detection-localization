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
    para_tr_data = ParasiteDatasetTrain(folder_to_read='../dataset/train_frames/')
    tr_data_loader = torch.utils.data.DataLoader(dataset=para_tr_data,
                                                 batch_size=32,  # binary number is good for programming
                                                 shuffle=True,
                                                 num_workers=15)

    para_tr_noaug_data = ParasiteDatasetTest(folder_to_read='../dataset/train_frames/')
    tr_noaug_data_loader = torch.utils.data.DataLoader(dataset=para_tr_noaug_data,
                                                       batch_size=32,
                                                       shuffle=False,
                                                       num_workers=15)

    para_ts_noaug_data = ParasiteDatasetTest(folder_to_read='../dataset/test_frames/')
    ts_noaug_data_loader = torch.utils.data.DataLoader(dataset=para_ts_noaug_data,
                                                       batch_size=32,
                                                       shuffle=False,
                                                       num_workers=15)

    para_val_noaug_data = ParasiteDatasetTest(folder_to_read='../dataset/val_frames/')
    val_noaug_data_loader = torch.utils.data.DataLoader(dataset=para_val_noaug_data,
                                                        batch_size=32,
                                                        shuffle=False,
                                                        num_workers=15)
    # set up for training
    folder = 'train_data'
    model_dir = "../pretrained_results/" + folder + "/saved_model/"
    suffix = '/train'
    gpu_id = 0

    model = models.vgg16(pretrained=True)
    # modify the last Fc layer into size 2 for positive sample and negative sample
    model.classifier[6] = nn.Linear(in_features=4096, out_features=2,
                                    bias=True)  #### what does this "6" of classifier[6] ??????
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001)

    for epoch in range(100):
        # print('\nepoch %d:' % epoch)
        # train the model
        train_model(model, tr_data_loader, criterion, optimizer, gpu_id=0)
        if epoch % 1 == 0:
            print('\nepoch %d:' % epoch)
            # validate the trained model for every 1 epoch
            val = validate_model(model, tr_noaug_data_loader, criterion, epoch,
                                 prediction_folder='../pretrained_results/train_data/', gpu_id=0)
            cvs_path = 'ep_' + str(epoch) + '_predictions.csv'
            arr = cvs_to_arr(cvs_file='../pretrained_results/train_data/' + cvs_path)
            metrics = get_metrics(arr)
            print('train_accuracy', metrics[0])

            # validate the test model for every 1 epoch
            val = validate_model(model, ts_noaug_data_loader, criterion, epoch,
                                 prediction_folder='../pretrained_results/test_data/', gpu_id=0)
            cvs_path = 'ep_' + str(epoch) + '_predictions.csv'
            arr = cvs_to_arr(cvs_file='../pretrained_results/test_data/' + cvs_path)
            metrics = get_metrics(arr)
            print('test_accuracy', metrics[0])

            # validate the validation model for every 1 epoch
            val = validate_model(model, val_noaug_data_loader, criterion, epoch,
                                 prediction_folder='../pretrained_results/val_data/', gpu_id=0)
            cvs_path = 'ep_' + str(epoch) + '_predictions.csv'
            arr = cvs_to_arr(cvs_file='../pretrained_results/val_data/' + cvs_path)
            metrics = get_metrics(arr)
            print('val_accuracy', metrics[0])

        if epoch % 10 == 0:
            # save model
            save_models(model, model_dir, epoch)

        if epoch == 30:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001)

        if epoch == 60:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001)

