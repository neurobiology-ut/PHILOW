# napari-PHILOW

[![License](https://img.shields.io/pypi/l/napari-PHILOW.svg?color=green)](https://github.com/neurobiology-ut/PHILOW/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-PHILOW.svg?color=green)](https://pypi.org/project/napari-PHILOW)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-PHILOW.svg?color=green)](https://python.org)
[![tests](https://github.com/neurobiology-ut/napari-PHILOW/workflows/tests/badge.svg)](https://github.com/neurobiology-ut/PHILOW/actions)
[![codecov](https://codecov.io/gh/neurobiology-ut/napari-PHILOW/branch/main/graph/badge.svg)](https://codecov.io/gh/neurobiology-ut/PHILOW)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-PHILOW)](https://napari-hub.org/plugins/napari-PHILOW)

# PHILOW <br>
***P***ython-based platform for ***h***uman-***i***n-the-***lo***op (HITL)  ***w***orkflow (PHILOW) <br>

PHILOW is an interactive deep learning-based platform for 3D datasets implemented on top of [napari](https://github.com/napari/napari)

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/stable/index.html
-->

## Installation

You can install `napari-PHILOW` via [pip]:

    pip install napari-PHILOW
    
or clone this repository   
then
```angular2
cd PHILOW
pip install -e .
```
    

## Usage

Launch napari 

```angular2
napari
```


#### load dataset


1) Plugins > napari-PHILOW > Annotation Mode

2) Select original dir : all slices must be in separate PNG and must be sequentially numbered (e.g. 000.png, 001.png ...)

3) Select mask dir : To resume from the middle of the annotation, specify here the name of the directory containing the mask image. The directory must contain the same number of files with the same name as the original image.   
 If you are starting a completely new annotation, you do not need to specify a directory. The directory for mask is automatically created and blank images are generated and stored.

4) Enter a name for the label or model you want to create (e.g. mito, cristae, ...)   
This name will be used as the directory name of the newly created mask dir if no mask dir is specified, 
and as the name of the csv file for training dataset management.

5) Check if you want to create new dataset (new model)
When checked, if there is already a csv file for training dataset management, a new csv file with one sequential number will be generated.

6) Start tracing


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
To train on your GPU machine (or with CPU), 

1) Plugins > napari-PHILOW > Trainer
   
2) Select original dir : all slices must be in separate PNG and must be sequentially numbered (e.g. 000.png, 001.png ...)   
   
3) Select labels dir : all label images should be named same as original images and contains data management csv file   
   
4) Select dir for save trained model   
   
5) Click on the "start training" button   

6) Dice score and dice loss are displayed. For more detail, check the command line for the progress of training. If you want to stop in the middle, click stop button.   
   
#### Predict
To predict labels on your machine,  

1) Plugins > napari-PHILOW > Predicter
   
2) Select original dir : all slices must be in separate PNG and must be sequentially numbered (e.g. 000.png, 001.png ...)   
   
3) (Optional) Select labels dir if you want to keep labels witch were used on training, and data management csv file   
   
4) Select model dir contains hdf5 file   
   
5) Select output dir for predicted labels   

6) Uncheck the box if you DO NOT want to use TAP (Three-Axis-Prediction)   
   
7) Click on the "predict" button  

8) Check the command line for the progress of prediction. If you want to stop in the middle, use ctrl+C.   

9) You can start the next round of annotation by selecting the merged_prediction directory as the mask dir in Annotation mode.

### Train and predict with Google Colab   
If you don't have a GPU machine, you can use Google Colab to perform GPU-based training and prediction for free.    

1) Open [train and predict notebook](https://github.com/neurobiology-ut/PHILOW/blob/develop/notebooks/train_and_pred_using_PHILOW.ipynb) and click "Open in Colab" button

2) You can upload your own dataset to train and predict, or try it on demo data   


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [GNU GPL v3.0] license,
"napari-PHILOW" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

# Authors <br>

Shogo Suga <br>
Hiroki Kawai <br>
<a href="http://park.itc.u-tokyo.ac.jp/Hirabayashi/WordPress/">Yusuke Hirabayashi</a> 


# How to Cite <br>
Shogo Suga, Koki Nakamura, Bruno M Humbel, Hiroki Kawai, Yusuke Hirabayashi, An interactive deep learning-based approach reveals mitochondrial cristae topologies
<a href="https://doi.org/10.1101/2021.06.11.448083">https://doi.org/10.1101/2021.06.11.448083</a>


```
@article {Suga2021.06.11.448083,
	author = {Suga, Shogo and Nakamura, Koki and Humbel, Bruno M and Kawai, Hiroki and Hirabayashi, Yusuke},
	title = {An interactive deep learning-based approach reveals mitochondrial cristae topologies},
	elocation-id = {2021.06.11.448083},
	year = {2021},
	doi = {10.1101/2021.06.11.448083},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2021/06/11/2021.06.11.448083},
	eprint = {https://www.biorxiv.org/content/early/2021/06/11/2021.06.11.448083.full.pdf},
	journal = {bioRxiv}
}
```

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
