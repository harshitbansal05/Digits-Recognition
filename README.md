# Digits-Recognition
*Digits Recognition* is an Android application that lets you click an image and detects the numbers present in it. It is accompanied by a TensorFlow implementation of **Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks**. It also comprises of a text detector **EAST: An Efficient and Accurate Scene Text Detector** for detecting digits in the images.

## Pre-requisites
* Python 3.6
* Tensorflow 1.10

## Setup
1. Clone the repository/source code. 

* To run the Android application, the model files for the text detector and number recognizer need to be placed in the `assets` folder in `app/src/main` directory of the root folder. The model files (in the protobuf format) can either be downloaded from Google Drive, or can be prepared by running the scripts for freezing graph of both the models.

* Freezing graph, in turn requires the trained weights and checkpoints. The final trained weights for both the models are available on Google Drive. Alternatively, the user can train the models and genarate final weights. To freeze the graph for the EAST model, navigate to the `east_tf` directory and run `python freeze_graph.py --checkpoint_path=./east_model`. Here the checkpoint path refers to the location of the trained weights for the model. This freezes the graph with the trained weights and variables in the same directory named `box_model_graph.pb`. 

* The graph for the number recognizer can also be freezed in a similar manner as above. For this, navigate to the `digit_classifier_tf` directory and perform the same steps. Finally, the two models can be placed in the assets folder and the app can be build. 

* To build the app, a jar(Java API) file and .so(C++ compiled) files for all the architectures need to be added to the `app/libs` directory. The files have been built and can be downloaded from Google Drive easily. They could also be built using Bazel, but it takes a large amount of time and a little hack is also needed to change the BUILD file for TensorFlow Android as it does not support TF operations like `tf.math.cos` and `tf.math.sin` which are needed for EAST model.

2. To train the number recognizer model, navigate to the `digit_classifier_tf` directory and run `python train_net.py`.

* It begins by downloading the Street View House Numbers (SVHN) train dataset in the `./svhn_data` directory. The SVHN train dataset consists of about 33,000 images and the extra dataset contains other 2,00,000 images of house numbers. Each image is resized to resolution *64x64* and the associated numbers and length are retrieved from the `digitStruct.mat` file. The resized image is converted to a numpy array and together with the two labels is serialized into a TF example and subsequently written to the TF-Records file.

* Finally, training begins. It is done for 50,000 iterations with a batch size of 32. The training checkpoints are saved in `/svhn_data/train_` directory. Also, a summary of the loss, gradient and weights is made and saved in `/svhn_data/summary` and can be used for visualization in Tensorboard. 

3. To evaluate the number recognizer model on the test dataset, run `python eval_net.py`. It performs the evaluation on the entire target dataset and prints the precision at the end. 

## Datasets
The Street View House Numbers (SVHN) Dataset is used as the dataset for training both the models. To train the models on the full dataset, download dataset [here](http://ufldl.stanford.edu/housenumbers/train.tar.gz).

## Contributing
Suggestions and pull requests are actively welcome.

## References
1. Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks ([Paper](https://arxiv.org/abs/1312.6082))
2. EAST: An Efficient and Accurate Scene Text Detector ([Paper](https://arxiv.org/abs/1704.03155v2))
3. EAST: An Efficient and Accurate Scene Text Detector ([Link](https://github.com/argman/EAST))
