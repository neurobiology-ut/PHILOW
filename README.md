# PHILOW <br>
***P***ython-based platform for ***h***uman-***i***n-the-***lo***op (HITL)  ***w***orkflow (PHILOW) <br>

## Overview 
PHILOW is an interactive deep learning-based platform for 3D datasets implemented on top of [napari](https://github.com/napari/napari) <br>

Features:

&nbsp;&nbsp;&nbsp;&nbsp; (1) generation of the ground truth data sets with annotation assistance, visualization and data management tools<br>
&nbsp;&nbsp;&nbsp;&nbsp; (2) model training and prediction <br>
&nbsp;&nbsp;&nbsp;&nbsp; (3) correcting the results of deep learning-based segmentations <br>
&nbsp;&nbsp;&nbsp;&nbsp; (4) iterate this process to get efficient and good results in a short time  <br>
&nbsp;&nbsp;  in a single user-friendly application running on Python.


## Installation
Clone this repository first.   
Then install requirements.
```angular2
git clone https://github.com/neurobiology-ut/PHILOW.git
cd PHILOW
pip install requirements.txt
```


## Usage
#### load dataset
```angular2
python launcher_simple.py
```
1) Select Loader

2) Select original dir : all slices must be in separate PNG and must be sequentially numbered (e.g. 000.png, 001.png ...)

3) Start new project?    
yes → do not need to select mask dir    
no → select saved labels dir    

4) Enter a name for the label or model you want to create (e.g. mito, cristae, ...)   
This name will be used as the directory name of the newly created mask dir if no mask dir is specified, 
and as the name of the csv file for training dataset management.

5) Check if you want to create new dataset (new model)
When checked, if there is already a csv file for training dataset management, a new csv file with one sequential number will be generated.

6) Launch napari!


#### create labels
Create a label with the brush function.
more information → https://napari.org/tutorials/fundamentals/labels.html

#### Orthogonal view
If you want to see orthogonal view, click on the location you want to see while holding down the Shift button.    
The image from xy, yz, and zx will be displayed on the right side of the screen.

#### Low confident layer
If you are in the second iteration and you are loading the prediction results, you will see a low confidence layer.    
This shows the area where the confidence of the prediction result is low.    
Use this as a reference for correction.   

#### Small object layer
We provide a small object layer to find small painted areas.   
This is a layer for displaying small objects.   
The slider widget on the left allows you to change the maximum object size to be displayed.   

#### save labels
If you want to save your label, click the "save" button on the bottom right.

#### select training dataset
We are providing a way to manage the dataset for use in training.   
If you want to use the currently displayed slice as your training data, click the 'Not Checked' button near the center left to display 'Checked'.


### Train and pred with your gpu machine
#### Train
To train on your GPU machine, open launcher at first.
```angular2
python launcher_simple.py
```
1) Select Trainer   
   
2) Select original dir : all slices must be in separate PNG and must be sequentially numbered (e.g. 000.png, 001.png ...)   
   
3) Select labels dir : all label images should be named same as original images and contains data management csv file   
   
4) Select dir for save trained model   
   
5) Click on the "start training" button   

6) Check the command line for the progress of training. If you want to stop in the middle, use ctrl+C.   
   
#### Predict
To predict labels on your machine, open launcher at first.   
```angular2
python launcher_simple.py
```
1) Select Predicter
   
2) Select original dir : all slices must be in separate PNG and must be sequentially numbered (e.g. 000.png, 001.png ...)   
   
3) (Optional) Select labels dir if you want to keep labels witch were used on training, and data management csv file   
   
4) Select model dir contains hdf5 file   
   
5) Select output dir for predicted labels   
   
6) Click on the "predict" button  

7) Check the command line for the progress of prediction. If you want to stop in the middle, use ctrl+C.    

### Train and predict with Google Colab   
If you don't have a GPU machine, you can use Google Colab to perform GPU-based training and prediction for free.    

1) Open [train and predict notebook](https://github.com/neurobiology-ut/PHILOW/blob/develop/notebooks/train_and_pred_using_PHILOW.ipynb) and click "Open in Colab" button

2) You can upload your own dataset to train and predict, or try it on demo data   


 
 　
# Authors <br>

Shogo Suga <br>
Hiroki Kawai <br>
<a href="http://park.itc.u-tokyo.ac.jp/Hirabayashi/WordPress/">Yusuke Hirabayashi</a> 


# References <br>
Shogo Suga, Koki Nakamura, Bruno M Humbel, Hiroki Kawai, Yusuke Hirabayashi, An interactive deep learning-based approach reveals mitochondrial cristae topologies
<a href="https://doi.org/10.1101/2021.06.11.448083">https://doi.org/10.1101/2021.06.11.448083</a>