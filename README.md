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
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 256, 256, 3  0           []                               
                                )]                                                                
                                                                                                  
 conv2d (Conv2D)                (None, 256, 256, 16  448         ['input_1[0][0]']                
                                )                                                                 
                                                                                                  
 dropout (Dropout)              (None, 256, 256, 16  0           ['conv2d[0][0]']                 
                                )                                                                 
                                                                                                  
 conv2d_1 (Conv2D)              (None, 256, 256, 16  2320        ['dropout[0][0]']                
                                )                                                                 
                                                                                                  
 max_pooling2d (MaxPooling2D)   (None, 128, 128, 16  0           ['conv2d_1[0][0]']               
                                )                                                                 
                                                                                                  
 conv2d_2 (Conv2D)              (None, 128, 128, 32  4640        ['max_pooling2d[0][0]']          
                                )                                                                 
                                                                                                  
 dropout_1 (Dropout)            (None, 128, 128, 32  0           ['conv2d_2[0][0]']               
                                )                                                                 
                                                                                                  
 conv2d_3 (Conv2D)              (None, 128, 128, 32  9248        ['dropout_1[0][0]']              
                                )                                                                 
                                                                                                  
 max_pooling2d_1 (MaxPooling2D)  (None, 64, 64, 32)  0           ['conv2d_3[0][0]']               
                                                                                                  
 conv2d_4 (Conv2D)              (None, 64, 64, 64)   18496       ['max_pooling2d_1[0][0]']        
                                                                                                  
 dropout_2 (Dropout)            (None, 64, 64, 64)   0           ['conv2d_4[0][0]']               
                                                                                                  
 conv2d_5 (Conv2D)              (None, 64, 64, 64)   36928       ['dropout_2[0][0]']              
                                                                                                  
 max_pooling2d_2 (MaxPooling2D)  (None, 32, 32, 64)  0           ['conv2d_5[0][0]']               
                                                                                                  
 conv2d_6 (Conv2D)              (None, 32, 32, 128)  73856       ['max_pooling2d_2[0][0]']        
                                                                                                  
 dropout_3 (Dropout)            (None, 32, 32, 128)  0           ['conv2d_6[0][0]']               
                                                                                                  
 conv2d_7 (Conv2D)              (None, 32, 32, 128)  147584      ['dropout_3[0][0]']              
                                                                                                  
 max_pooling2d_3 (MaxPooling2D)  (None, 16, 16, 128)  0          ['conv2d_7[0][0]']               
                                                                                                  
 conv2d_8 (Conv2D)              (None, 16, 16, 256)  295168      ['max_pooling2d_3[0][0]']        
                                                                                                  
 dropout_4 (Dropout)            (None, 16, 16, 256)  0           ['conv2d_8[0][0]']               
                                                                                                  
 conv2d_9 (Conv2D)              (None, 16, 16, 256)  590080      ['dropout_4[0][0]']              
                                                                                                  
 conv2d_transpose (Conv2DTransp  (None, 32, 32, 128)  131200     ['conv2d_9[0][0]']               
 ose)                                                                                             
                                                                                                  
 concatenate (Concatenate)      (None, 32, 32, 256)  0           ['conv2d_transpose[0][0]',       
                                                                  'conv2d_7[0][0]']               
                                                                                                  
 conv2d_10 (Conv2D)             (None, 32, 32, 128)  295040      ['concatenate[0][0]']            
                                                                                                  
 dropout_5 (Dropout)            (None, 32, 32, 128)  0           ['conv2d_10[0][0]']              
                                                                                                  
 conv2d_11 (Conv2D)             (None, 32, 32, 128)  147584      ['dropout_5[0][0]']              
                                                                                                  
 conv2d_transpose_1 (Conv2DTran  (None, 64, 64, 64)  32832       ['conv2d_11[0][0]']              
 spose)                                                                                           
                                                                                                  
 concatenate_1 (Concatenate)    (None, 64, 64, 128)  0           ['conv2d_transpose_1[0][0]',     
                                                                  'conv2d_5[0][0]']               
                                                                                                  
 conv2d_12 (Conv2D)             (None, 64, 64, 64)   73792       ['concatenate_1[0][0]']          
                                                                                                  
 dropout_6 (Dropout)            (None, 64, 64, 64)   0           ['conv2d_12[0][0]']              
                                                                                                  
 conv2d_13 (Conv2D)             (None, 64, 64, 64)   36928       ['dropout_6[0][0]']              
                                                                                                  
 conv2d_transpose_2 (Conv2DTran  (None, 128, 128, 32  8224       ['conv2d_13[0][0]']              
 spose)                         )                                                                 
                                                                                                  
 concatenate_2 (Concatenate)    (None, 128, 128, 64  0           ['conv2d_transpose_2[0][0]',     
                                )                                 'conv2d_3[0][0]']               
                                                                                                  
 conv2d_14 (Conv2D)             (None, 128, 128, 32  18464       ['concatenate_2[0][0]']          
                                )                                                                 
                                                                                                  
 dropout_7 (Dropout)            (None, 128, 128, 32  0           ['conv2d_14[0][0]']              
                                )                                                                 
                                                                                                  
 conv2d_15 (Conv2D)             (None, 128, 128, 32  9248        ['dropout_7[0][0]']              
                                )                                                                 
                                                                                                  
 conv2d_transpose_3 (Conv2DTran  (None, 256, 256, 16  2064       ['conv2d_15[0][0]']              
 spose)                         )                                                                 
                                                                                                  
 concatenate_3 (Concatenate)    (None, 256, 256, 32  0           ['conv2d_transpose_3[0][0]',     
                                )                                 'conv2d_1[0][0]']               
                                                                                                  
 conv2d_16 (Conv2D)             (None, 256, 256, 16  4624        ['concatenate_3[0][0]']          
                                )                                                                 
                                                                                                  
 dropout_8 (Dropout)            (None, 256, 256, 16  0           ['conv2d_16[0][0]']              
                                )                                                                 
                                                                                                  
 conv2d_17 (Conv2D)             (None, 256, 256, 16  2320        ['dropout_8[0][0]']              
                                )                                                                 
                                                                                                  
 conv2d_18 (Conv2D)             (None, 256, 256, 2)  34          ['conv2d_17[0][0]']              
                                                                                                  
==================================================================================================
Total params: 1,941,122
Trainable params: 1,941,122
Non-trainable params: 0
__________________________________________________________________________________________________
