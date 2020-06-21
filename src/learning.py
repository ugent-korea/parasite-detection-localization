import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
# from torchsummary import summary

from thesis_dataset import ParasiteDatasetTrain
from assistant import write_to_csv


def train_model(model, data_train, criterion, optimizer, gpu_id=1):
    """
    Training the model

    model: the model to be trained
    data_train: training dataset
    criterion: cost function
    optimizer: update weights
    gpu_id: ID of gpu

    return -> outputs
    """
    model.train()
    criterion = criterion.cuda(gpu_id)
    model.cuda(gpu_id)
    for index, (images, labels) in enumerate(data_train):
        # put in variable
        images = Variable(images.cuda(gpu_id))                  # Sending images to gpu server
        labels = Variable(labels.cuda(gpu_id))
        optimizer.zero_grad()                                   # Clear old gradients (otherwise, gradients accumulate)
        outputs = model(images)                                 # Forward pass
        loss = criterion(outputs, labels)                       # Calculate loss
        loss.backward()                                         # Backward pass
        optimizer.step()                                        # Update weights
        # print(outputs)
        # print(outputs.size())


def validate_model(model, data_val, criterion, epoch=0, prediction_folder='../results/', gpu_id=1):
    """
    Validate(evaluate) a model with only forward pass and save into cvs form

    data_val: data for validation
    criterion: cost(loss) function
    epoch:
    prediction_folder: place for cvs files will be saved

    return -> generate a __.cvs into specific folder
    """
    model.eval()
    model.cuda(gpu_id)
    for index, (images, labels) in enumerate(data_val):
        with torch.no_grad():
            # put in variable
            images = Variable(images.cuda(gpu_id))
            labels = Variable(labels.cuda(gpu_id))
            output = model(images)                                 # forward pass
            loss = criterion(output, labels)                       # Calculate loss
            # print('loss', loss)
            pred = torch.argmax(output, dim=1)                     # prediction
            # softmax_score = softmax(output)
            # generate folders if does not exist
            if not os.path.exists(prediction_folder):
                os.makedirs(prediction_folder)

            # Save logits ([0]:true_pred, [1]:output[0], [2]:output[1])
            output = output.cpu()
            for label_item, out_item in zip(labels, output):
                write_to_csv(prediction_folder + 'ep_' + str(epoch) + '_logit.csv',
                             [label_item.item(), out_item[0].item(), out_item[1].item()])
            # print('{0}', str(type(loss)))

            # Save Loss ([0]:loss)
            file_to_write = open(prediction_folder + 'ep_' + str(epoch) + '_loss.csv', 'a')
            file_to_write.write(str(loss.item()) + ',')
            file_to_write.write('\n')
            file_to_write.close()

            # Save Predictions ([0]:true_pred, [1]:predicted]
            for true_out, pred_out in zip(labels, pred):
                write_to_csv(prediction_folder + 'ep_' + str(epoch) + '_predictions.csv',
                             [true_out.item(), pred_out.item()])


if __name__ == "__main__":
    # path for input
    para_dataset_tr = ParasiteDatasetTrain(folder_to_read='../dataset/train_data/')
    para_dataset_loader = torch.utils.data.DataLoader(dataset=para_dataset_tr,
                                                      batch_size=20,
                                                      shuffle=False)
    model = models.vgg16()
    # modify the last Fc layer into size 2 for positive sample and negative sample
    model.classifier[14] = nn.Linear(in_features=4097000, out_features=2, bias=True)
    # print(summary(model, input_size=(3,224,224)))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # train_model(model, para_dataset_loader, criterion, optimizer)
    validate_model(model, para_dataset_loader, criterion, epoch=0, gpu_id=1)
