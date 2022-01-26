# Winstars Technology. Test task for Internship in Data Science.
# Project goal
The goal of the test project is to build a semantic segmentation model. Prefered tools and notes: tf.keras, Unet architecture for neural network, dice score, python. 
# Configuration instructions 
Before you can run this scipts, you need to be sure, that every needed library from requirenments.txt is installed. And you can run it.
# Installation instructions 
Download dataset, if you want to train this model, or use pre trained model in model directory.
# A list of files included 
        Exploratory data analysis of the dataset.ipynb
        README.md
        inference.py
        requirements.txt
        train.py
        model\
                model_4.h5
# Description of the solution
For starters we find all images with amount of ships greater than 13, so that we can train model more quickly on them. Then we patchify those images (1 x 768x768 => 9 x 256x256), to get larger dataset and give the model more variety of situation. After that we create UNet model with 9 layers, where 5 of them are for contraction and the other 4 are for expansion. Then define function for dice score loss and Jacard coefficient for model metrics. Upgrade dice score loss function (no particular reason), compile model and start training. To get more or less good results the minimum amount of epochs should be around 30, after 50 epochs training proccess is very long (you may not see any differencies between 50 and 70 epochs of training). After training is done, you may save the model.
Inference is not very different in that aspect. Firstly we choose any image we want to use for prediction. This image will be pathified into 9 pieces, after that every piece will go through the model and be evaluated. In the end, predicted masks would be merged in one and displayed with original image and mask.
