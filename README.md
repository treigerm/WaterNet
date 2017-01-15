# DeepWater

This is a program which learns to identify water in satellite images.

## Installation

For running the program yourself you will need Docker, some actual satellite imagery and shapefiles to create the labels.

### Getting the data

I got my satellite imagery from [USGS](https://earthexplorer.usgs.gov/) (you will need to register to download the data). The shapefiles are from [Geofabrik](http://download.geofabrik.de/). I've provided links on which data I used for training my classifier in [Downloads](##Downloads).

### Running the program

First you will need to install Docker. You can find instructions on how to do this on their [homepage](https://www.docker.com/products/overview). 

Then build and start the container with 
```
$ docker build -t deep_water .
$ docker run -v /path/to/data:/data -it deep_water /bin/bash
```
Here `/path/to/data` is the path to the data directory described in [Data directory](##Data directory).

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

## Downloads
