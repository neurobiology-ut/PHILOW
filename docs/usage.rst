.. usage::

Usage
=====


Launch napari
-------------

.. code-block::

    napari


Load dataset
------------

Plugins > napari-PHILOW > Annotation Mode
"""""""""""""""""""""""""""""""""""""""""

.. image:: images/image_001.jpeg

Select original dir : all slices must be in separate PNG and must be sequentially numbered (e.g. 000.png, 001.png ...)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Select mask dir
"""""""""""""""

| To resume from the middle of the annotation, specify here the name of the directory containing the mask image. 
The directory must contain the same number of files with the same name as the original image.
| If you are starting a completely new annotation, you do not need to specify a directory. 
The directory for mask is automatically created and blank images are generated and stored.