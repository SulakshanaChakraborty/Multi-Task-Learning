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

Folder Structure:

- The MTL folder structure consists of files: Instructions.txt, attention.py, cw2_main.py,
data_loader_canny.py, denoising_loader.py, displaying.py, generate_noisy_data.py, lab_loader.py,
load_data.py, losses.py, losses_denoising.py, metrics.py, model_utils.py, save_lab_images.py,
test_model.py, train_canny.py, train_color.py, train_denoising.py, train_model.py and pt_networks
subfolder. The pt_networks subfolder contains the networks for used to create the trained models.

Models:

- The following models can be found in 'model_type_list' list of cw2_main.py :

'Segnet-1task-untrained', 'Segnet-1task', 'MTL-Segnet-untrained', 'MTL-Segnet', 'MTL-Attention', 
'MTL-Attention-with-colorization', 'MTL-Attention-with-denoising', 'MTL-Attention-with-canny',
'MTL-Attention-without-bbox', 'MTL-Attention-without-classification', 'MTL-segnet-with-canny',
'MTL-segnet-with-colorization'

- To set the model in the command line when running cw2_main, the following command should be used and
the name of the model added to the end after a space like so: 
'python cw2_main.py -m {model name}'

eg: for the MTL attention model with colorization: 
'python cw2_main.py -m MTL-Attention-with-colorization'

Experiments 

Examples for experimenting with other different models (for the testing mode with all default arguments):
'python cw2_main.py -m Segnet-1task-untrained'
'python cw2_main.py -m Segnet-1task'
'python cw2_main.py -m MTL-Segnet-untrained'
'python cw2_main.py -m MTL-Segnet'
'python cw2_main.py -m MTL-Attention'
'python cw2_main.py -m MTL-Attention-with-colorization'
'python cw2_main.py -m MTL-Attention-with-denoising'
'python cw2_main.py -m MTL-Attention-with-canny'
'python cw2_main.py -m MTL-Attention-without-bbox'
'python cw2_main.py -m MTL-Attention-without-classification'
'python cw2_main.py -m MTL-segnet-with-canny'
'python cw2_main.py -m MTL-segnet-with-colorization'


Data Loading: 

- The data required for training, validation and testing is initially loaded, using the resepective file paths
which should be kept constant in the cw2_main.py file as they are hard coded throughout the repository
files (ie: 'data/train/' for training, 'data/validation/' for the validation and 'data/test/' for testing data.

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


Colorisation Scripts:

- An additional colorisation script lab_loader.py was added to convert the RGB colour space to the LAB colour
space. This file must be ran before the running of cw2_main.py and training/testing of a model.  

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

