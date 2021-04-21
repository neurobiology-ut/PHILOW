# Philis <br>
***P***ython-based ***h***uman-***i***n-the-***l***oop (HITL) ***i***terative ***s***egmentation platform (Philis) <br>

## Overview 
Philis integrated <br>
&nbsp;&nbsp;&nbsp;&nbsp; (1) generation of the ground truth data sets, <br>
&nbsp;&nbsp;&nbsp;&nbsp; (2) correcting the results of deep learning-based segmentations, <br>
&nbsp;&nbsp;&nbsp;&nbsp; (3) post-processing, such as restrictive thresholding, object quantification, and structure visualization, <br>
&nbsp;&nbsp;  in a single user-friendly application running on Python.

## Installation
Clone this repository first.   
Then install requirements.
```angular2
git clone https://github.com/neurobiology-ut/Philis.git
cd Philis
pip install requirements.txt
```

## Usage
#### load dataset
```angular2
python napari_view_simple.py
```
1) select ModificationsLoader

2) select original dir : all slices must be in separate PNG and must be sequentially numbered (e.g. 000.png, 001.png ...)

3) start new project?
yes → do not need to select mask dir
no → select saved labels dir

4) enter a name for the label or model you want to create (e.g. mito, cristae, ...)   
This name will be used as the directory name of the newly created mask dir if no mask dir is specified, 
and as the name of the csv file for training dataset management.

5) check if you want to create new dataset (new model)
When checked, if there is already a csv file for training dataset management, a new csv file with one sequential number will be generated.

6) launch napari!

#### create labels
Create a label with the brush function.
more information → https://napari.org/tutorials/fundamentals/labels.html

#### Orthogonal view
If you want to see orthogonal view, click on the location you want to see while holding down the Shift button.    
The image from xy, yz, and zx will be displayed on the right side of the screen.

#### save labels
If you want to save your label, click the "save" button on the bottom right.

#### select training dataset
We are providing a way to manage the dataset for use in training.   
If you want to use the currently displayed slice as your training data, click the 'Not Checked' button near the center left to display 'Checked'.


# Authors <br>

Shogo Suga <br>
Hiroki Kawai <br>
<a href="http://park.itc.u-tokyo.ac.jp/Hirabayashi/WordPress/">Yusuke Hirabayashi</a> 
