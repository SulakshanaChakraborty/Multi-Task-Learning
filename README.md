## Enhancing Multi-Task Learning for Image Segmentation using soft-attention blocks and supervised auxiliary tasks 

####  For setting up the environment

* Clone the repository
* Create an enviroment using the requirements.txt file

#### For Running Inference 

run **cw2_main.py -ts 'y'**

This would result in running the default MTL-Attention model, to run other pretrained model pass the model name as shown:

'-m'  : model type (deafult is set to 'MTL-Attention') <br>

The different pre-trained models present are:

* **'Segnet-1task-untrained'** : Vanilla Segnet model which outputs the segmentaion mask, without any pre-trained weights

* **'MTL-Segnet-untrained'** : Multi task learning Segnet model with Bouding Box Regression, Segmentation and Classification tasks, without any pre-trained weights

* **'Segnet-1task'** : Vanilla Segnet model which outputs the segmentaion mask,with pre-trained weights in encoder

* **'MTL-Segnet'** : Multi task learning Segnet model with Bouding Box Regression, Segmentation and Classification tasks, with pre-trained weights in encoder

* **'MTL-Attention'** : Soft Attention masks applied to the MTL Segnet model, with pre-trained weights in encoder

* **'MTL-Attention-without-classification'** : MTL Attention model with only Bounding Box Regression and Segmentaion, with pre-trained weights in encoder

* **'MTL-Attention-without-bbox'** : MTL Attention model with only Bounding Box Regression and Segmentaion, with pre-trained weights in encoder

* **'MTL-Attention-with-colorization'** : MTL Attention model with added self-supervised task of colorization, with pre-trained weights in encoder

* **'MTL-Attention-with-canny'**: MTL Attention model with added self-supervised task of canny edge detection, with pre-trained weights in encoder

* **'MTL-Attention-with-denoising'** : MTL Attention model with added self-supervised task of denoising, with pre-trained weights in encoder

❗Please Note: Before running the 'MTL-Attention-with-colorization' model an additional colorisation script lab_loader.py has to be run. This script converts the RGB colour images to the LAB colour images. This file must be ran before the running of cw2_main.py and training/testing of a model.  



--------------------------------------------------------------------------------------------------------------
Data Loading: 

- The data required for training, validation and testing is initially loaded, using the resepective file paths
which should be kept constant in the cw2_main.py file as they are hard coded throughout the repository
files (ie: 'data/train/' for training, 'data/validation/' for the validation and 'data/test/' for testing data.)

---------------------------------------------------------------------------------------------------------------
#### For training models

run **cw2_main.py -tr 'y'**

: <br>
'-m'  : model type (deafult is set to 'MTL-Attention') <br>
'-e'  : number of epochs (default 30)<br>
'-b'  : mini batch size (default 5)<br>
'-d'  : 'cpu' or 'cuda' device to run the code on<br>

- To train the default model with the other arguments set to the defaults, a user has to type 
'cw2_main.py -tr y' in the terminal. The default setting for the model is 'MTL-Attention' and this
be changed by adding the '-m' statement followed by the desired model name. The device on which the
program can be run on can be adjusted by using the '-d' statement followed by the name of the device
ie: 'CPU' or 'cuda', of which 'cuda' is the default. The mini batch size can be adjusted by using the 
'-b' followed by an integer (the default mini batch size is set to 5). Similarly, the number of epochs can 
be adjusted using the '-e' statement and then the chosen number (default is set to 30 epochs). 
- To execute the command the user has to click enter after adding their arguments. 
- To get help for a specific statement for an argument the user has to write the argument statement
followed by the '-h'. For example if the user wants to find out more about the '-d' statement, then 
they can use '-d -h' which will display a help message 'which device would you like to run on (cuda/cpu)'
- The ability to change model using '-m' allows for different experiments to be ran using the same
'cw2_main.py' file without having to modify it.

---------------------------------------------------------------------------------------------------------
Testing:

- The default running mode setting for the cw2_main.py is testing as in the ArgumentParser the default
argument for testing is set to 'y'. This means that to test the default 'MTL-Attention' model, the user
simply has to run 'python cw2_main.py' in the command without the addition of any other arguments.
- To perform testing on different models, the '-m' statement should be used followed by the name of the
chosen model. The process of changing the arguments for the testing mode follows the same methodology 
as described in the Training section of this document. 
 
















© 2022 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About

