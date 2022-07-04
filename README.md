# Multi-Task Learning for Image Segmentation using attention blocks

This repo contains the code, data and pretrained models for "Enhancing Multi Task Learning for Image Segmentaion using soft attention blocks and self-supervised auxilary tasks.  <br>

The motivation was to check the effectiveness of various MTL architecture against the baseline SegNet model. MTL model with auxilary tasks of Boundary box detection and Classification was implemented. Attention masks were added to the MTL model (MTAN). Experiments were carried out by adding attention masks and additional self-supervised auxilary tasks of edge detection, colourisation and denosing. The full report can be viewed [here](https://github.com/SulakshanaChakraborty/Multi-Task-Learning/blob/main/MTL-Report.pdf).

#### MTAN model architecture
<p>
<img src="MTAN.jpeg" alt="drawing" width="600" height = "300"/>
</p>

### Set up environment 
* Clone the repository
* Create an enviroment using the requirements.txt file

### Run Inference/Traing

Inference:
```python 
python cw2_main.py -m 'MTL-Segnet' -d 'cpu' -e '50' -b '10' -tr 'y' -ts 'n'
```
F refere to instructions.txt 

### Dataset 

* To run inference on the MTL-attention model run : cw2_main.py -ts 'y'

For further deatils on training and running inference on different pre-trained models, please refer to instructions.txt.





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

