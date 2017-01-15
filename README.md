# DeepWater

Using publicly available satellite imagery and OSM data we train a convolutional neural net to predict water occurrences in satellite images.

![Example output](/../images/imgs/budapest.jpg)

## Installation

For running the program yourself you will need Docker, some actual satellite imagery and corresponding shapefiles to create the labels.

### Getting the data

I got my satellite imagery from [USGS](https://earthexplorer.usgs.gov/) (you will need to register to download the data). The shapefiles are from [Geofabrik](http://download.geofabrik.de/). I've provided links on which data I used for training my classifier in [Downloads](#downloads).

### Running the program

First you will need to install Docker. You can find instructions on how to do this on their [homepage](https://www.docker.com/products/overview). 

Then build and start the container with 
```
$ docker build -t deep_water .
$ docker run -v /path/to/data:/data -it deep_water /bin/bash
```
Here `/path/to/data` is the path to the data directory described in [Data directory](#data-directory).

## Data directory

I tried to follow Ali Eslami's great [blog post](http://arkitus.com/patterns-for-research-in-machine-learning/) about structuring your Machine Learning projects. Therefore I have a data directory which is split up into input, working and output directories. So my data directory looks like this:
```
/data
  /input
    /{satellite provider name}
    /Shapefiles
  /working
    /models
    /train_data
      /labels_images
      /tiles
      /water_bitmaps
      /WGS84_images
  /output
    /{model_id}
    /tensorboard
```

## Things left to do

## Downloads

## Acknowledgements

[DeepOSM](https://github.com/trailbehind/DeepOSM) from [TrailBehind](https://github.com/trailbehind) helped a lot to get started on this project. It also links to several useful articles and related projects. 

Volodymyr Mnih's PhD thesis [Machine Learning for Aerial Image Labeling](https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf) was a great read and helped a lot to build ConvNet architecture.
