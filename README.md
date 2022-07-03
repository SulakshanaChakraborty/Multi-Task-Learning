## Enhancing Multi-Task Learning for Image Segmentation using soft-attention blocks and self-supervised auxiliary tasks 

Running Instructions 

Skip to content
Search or jump to…
Pull requests
Issues
Marketplace
Explore
 
@michalpaw18 
ArnabPushilal
/
MLT
Private
Code
Issues
Pull requests
Actions
Projects
Security
Insights
MLT/Instructions.txt
@michalpaw18
michalpaw18 Update Instructions.txt
Latest commit 7d23f72 17 minutes ago
 History
 1 contributor
96 lines (64 sloc)  4.72 KB
   
Instructions for running the code:

- Note customisation of the execution of the program takes place in the terminal, instead of
manually changing the main repository file. For this purpose, an ArgumentParser was utilised to
have command line arguments during the running of the program in the terminal. The information in
this document primarily focusses on the cw2_main.py file of the repository.
-----------------------------------------------------------------------------------------
Folder Structure:

- The MTL folder structure consists of files: Instructions.txt, attention.py, cw2_main.py,
data_loader_canny.py, denoising_loader.py, displaying.py, generate_noisy_data.py, lab_loader.py,
load_data.py, losses.py, losses_denoising.py, metrics.py, model_utils.py, save_lab_images.py,
test_model.py, train_canny.py, train_color.py, train_denoising.py, train_model.py and pt_networks
subfolder. 
The pt_networks subfolder contains the networks for used to create the trained models.

The data subfolder is where the data is placed - 

the train data is in data\train
the validation data is in data\val
the hold out test data is in data\test
----------------------------------------------------------------------------------------------
Running the experiments:

Command line arguments:
'-m'  : model type to experiment on (deafult is set to 'MTL-Attention')
'-e'  : number of epochs (default 30)
'-b'  : mini batch size (default 5)
'-tr' : 'y' (yes) or 'n' (no) for training the model (default is 'n')
'-ts' : 'y' (yes) or 'n' (no) for testing the model (default is 'y')
'-d'  : 'cpu' or 'cuda' device to run the code on

Examples for experimenting with other different models:
python cw2_main.py -m 'MTL-Segnet' -d 'cpu' -e '50' -b '10' -tr 'y' -ts 'n'

following is the list of model type:

'Segnet-1task-untrained' : Vanilla Segnet model which outputs the segmentaion mask, without any pre-trained weights

'MTL-Segnet-untrained' : Multi task learning Segnet model with Bouding Box Regression, Segmentation and Classification tasks, without any pre-trained weights

'Segnet-1task': Vanilla Segnet model which outputs the segmentaion mask,with pre-trained weights in encoder

'MTL-Segnet': Multi task learning Segnet model with Bouding Box Regression, Segmentation and Classification tasks, with pre-trained weights in encoder

'MTL-Attention' : Soft Attention masks applied to the MTL Segnet model, with pre-trained weights in encoder

'MTL-Attention-without-classification': MTL Attention model with only Bounding Box Regression and Segmentaion, with pre-trained weights in encoder

'MTL-Attention-without-bbox': MTL Attention model with only Bounding Box Regression and Segmentaion, with pre-trained weights in encoder

'MTL-Attention-with-colorization': MTL Attention model with added self-supervised task of colorization, with pre-trained weights in encoder

'MTL-Attention-with-canny': MTL Attention model with added self-supervised task of canny edge detection, with pre-trained weights in encoder

'MTL-Attention-with-denoising': MTL Attention model with added self-supervised task of denoising, with pre-trained weights in encoder

Please Note: Before running the 'MTL-Attention-with-colorization' model an additional colorisation script lab_loader.py has to be run. This script converts the RGB colour images to the LAB colour images. This file must be ran before the running of cw2_main.py and training/testing of a model.  



--------------------------------------------------------------------------------------------------------------
Data Loading: 

- The data required for training, validation and testing is initially loaded, using the resepective file paths
which should be kept constant in the cw2_main.py file as they are hard coded throughout the repository
files (ie: 'data/train/' for training, 'data/validation/' for the validation and 'data/test/' for testing data.)

---------------------------------------------------------------------------------------------------------------
Training:

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

