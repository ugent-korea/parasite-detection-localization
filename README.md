# Parasite detection and localization
This is the BA4 dissertation in Ghent university.

## Abstract
The goal of this project is the correct classification using Convolutional Neural Network and weakly-supervised localization using visualization. VGG-16 Architecture is used as CNN architecture and Gradient-weighted Class Activation Map (Grad-CAM) for visualization. The data for this project is the microscopy images of thick blood smears that were extracted from Trypanosoma parasites infected and non-infected mice. Hence, the positive samples are images containing parasites and the negative samples are images not-containing parasites. We conclude from experiments that the model can correctly classify the images up to 99% accuracy but not perfectly localize the parasites. The visualization showed that the model more focus on the global structures such as the clustering of blood cell rather than the parasites themselves.

## VISUALIZATION
Here is the final result which is the most reasonable among a number of trials with the different tuning of data or model parameters.

images/example_image.jpeg

