# WaterNet

Using publicly available satellite imagery and OSM data we train a convolutional neural net to predict water occurrences in satellite images. WaterNet is not supposed to achive state of the art results but rather to be a simple example of a machine learning technique applied to geospatial data.

![Figure 1](/../images/imgs/figure_1.jpg)

The picture is part of an example output of the classifier. The green parts are true positives, the red parts are false positives, the blue parts are false negatives and the rest are true negatives. With only 20 minutes of training I was able to train a classifier which has 96.38 % accuracy, 74.2 % precision and 49.04 % recall. As mentioned above, my goal was not to find the best classifier for this task but more to give an example of a simple architecture which allows to train a neural net on satellite data. I am certain that with a little bit more work it will be possible to create a significantly better classifier.

## Functionality

WaterNet can do the following things:

- Train a neural network with GeoTIFF satellite images and OSM shapefiles
- Create a visualisation of the neural nets predictions on the test dataset
- Evaluate the neural net by calculating accuracy, precision and recall, as well as a precision-recall curve
- Print hyperparameters of the neural net to a .txt file
- Use tensorboard for logging
- Save models with weights to disk for later usage
- Choose between datasets
- Run different computations seperately e.g. we can decide to only preprocess the data or only evaluate an alreaday trained model

## Installation

For running the program yourself you will need some actual satellite imagery and corresponding shapefiles to create the labels. For convenience I also included a Dockerfile.

### Getting the data

I got my satellite imagery from [USGS](https://earthexplorer.usgs.gov/) (you will need to register to download the data). The shapefiles are from [Geofabrik](http://download.geofabrik.de/). I've provided information on which data I used for training my classifier in [Downloads](#downloads).

### Running it

First you will need to install Docker. You can find instructions on how to do this on their [homepage](https://www.docker.com/products/overview). 

Then build and start the container with 
```
$ docker build -t water_net .
$ docker run -v /path/to/data:/data -it water_net /bin/bash
```
Here `/path/to/data` is the path to the data directory described in [Data directory](#data-directory). In the container simply type
```
$ waterNet -h
```
to get information about how to use the program. If you haven't already created all the folders in the `working` and `output` directories you will also want to run
```
$ waterNet --setup
```


## Data directory

I tried to follow Ali Eslami's great [blog post](http://arkitus.com/patterns-for-research-in-machine-learning/) about structuring your Machine Learning projects. Therefore I have a data directory which is split up into input, working and output directories. So my data directory looks like this:
```
/data
  /input
    /{satellite provider name}
    /Shapefiles
  /working
    /models
      /{model_id}
    /train_data
      /labels_images
      /tiles
      /water_bitmaps
      /WGS84_images
  /output
    /{model_id}
    /tensorboard
```

All the metrics and hyperparameters of a model are stored in `/output/{model-id}` model weights are stored under `/working/models/{model-id}`. The model ID is a string consisting of the current timestamp, the dataset that is used and the neural net architecture. `/output/tensorboard` contains the logs for tensorboard. The last repository which might be of interest is `/working/train_data/labels_images` which contains the visualisations of the water polygons for a satellite images. These images will be created if you run the script with the `-v` tag. The remaining directories in `/working/train_data` are used as caches for the preproccesing of the data.

## Downloads

### Shapefiles

Downloads of shapefiles are provided [here](http://download.geofabrik.de/). These shapefiles do not contain large water bodies like the oceans. You can find shapefiles for the ocean polygons [here](http://openstreetmapdata.com/data/water-polygons). This are the regions I downloaded from Geofabrik (all in Europe):
```
Netherlands
England
Bayern (subregion of Germany)
Nordrhein-Westfalen (subregion of Germany)
Hungary
Nord-est (subregion of Italy)
```
After you downloaded the shapefiles, place the expanded zip files in the `Shapefiles` directory and remove the ".shp" extension from the folder.

### Satellite images

You can download the satellite imagery from the [USGS Earth Explorer](https://earthexplorer.usgs.gov/). The following are the entity IDs of the images I used. To find images by their ID first select the right dataset (in our case Sentinel-2) and then go to "Additional criteria". Here are the IDs I used:
```
S2A_OPER_MSI_L1C_TL_SGS__20161204T105758_20161204T143433_A007584_T32ULC_N02_04_01
S2A_OPER_MSI_L1C_TL_SGS__20160908T110617_20160908T161324_A006340_T31UFU_N02_04_01
S2A_OPER_MSI_L1C_TL_SGS__20160929T103545_20160929T154211_A006640_T32UPU_N02_04_01
S2A_OPER_MSI_L1C_TL_SGS__20160719T112949_20160719T165219_A005611_T30UVE_N02_04_01
S2A_OPER_MSI_L1C_TL_SGS__20161115T101822_20161115T171903_A007312_T32TQR_N02_04_01
L1C_T30UXC_A007999_20170102T111441
S2A_OPER_MSI_L1C_TL_SGS__20161129T100308_20161129T134154_A007512_T33TYN_N02_04_01
```
After you downloaded the image, place them in a directory named `Sentinel-2` under the `input` directory.
Please take a look at the config.py file to see which shapefiles belong to which satellite images.

## Pull requests welcome

I am new to geospatial analysis and writing machine learning code, so if you have ideas about how to improve this program you are more than welcome to open an issue or create a pull request!

## Acknowledgements

[DeepOSM](https://github.com/trailbehind/DeepOSM) from [TrailBehind](https://github.com/trailbehind) helped a lot to get started on this project. It also links to several useful articles and related projects. 

Volodymyr Mnih's PhD thesis [Machine Learning for Aerial Image Labeling](https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf) was a great read and helped a lot to build the ConvNet architecture.
