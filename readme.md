# TableTrainNet
## A simple project for training and testing table recognition in documents.
This project was developed to train a neural network to detect tables inside documents.
Once the model is trained, it can be used in
[IntelligentOCR](https://github.com/nevepura/IntelligentOCR)
to detect tables.

## General overview
The project uses the pre-trained neural network 
[offered](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
by Tensorflow. In addition, a 
[config](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)
file was used, according to the choosen pre-trained model, to train with 
[object detections tensorflow API](https://github.com/tensorflow/models/tree/master/research/object_detection#tensorflow-object-detection-api)

The datasets was taken from:
* [TableBank](https://github.com/doc-analysis/TableBank).
Only part of the dataset was used. Which part? The one contained in the folder "TableBank/TableBank_data/Detection_data", which had a proper annotation. The pictures from "Recognition_data" had no convenient annotations, since they are intended to understand the structure of a table, and not its frame. 
Note that the dataset is not freely available, because copyright. If you want it, apply to request it on TableBank github page.

## Required libraries
Before we go on make sure you have everything installed to be able to use the project:
* Python 3
* [Tensorflow](https://www.tensorflow.org/) (tested on r1.8)
* Its [object-detection API](https://github.com/tensorflow/models/tree/master/research/object_detection#tensorflow-object-detection-api)
(remember to install COCO API. If you are on Windows see at the bottom of the readme)
* Pillow
* opencv-python
* pandas
* pyprind (useful for process bars)

Notice: to complete the installation of the Coco API, you will have to clone the
[tensorflow/models](https://github.com/tensorflow/models).
This will also be useful later to train your model, since all the training files lie in tensorflow/models.


## Project pipeline
The project is made up of different parts that acts together as a pipeline.

###Disclaimer: on pictures and annotations
The project hosted here contains the annotations, but not the pictures.
It won't work unless you load the correct pictures, which are contained in the TableBank dataset, 
in the folder `TableBank_data/Detection_data`.
In alternative, you can load your own dataset and your own annotations.
Make sure that the annotations have the same format as the current ones.
An example here:
```angular2html
<?xml version='1.0' encoding='UTF-8'?>
<document filename="1401.0007_15">
  <tableRegion>
    <Coords points="85,396 510,396 85,495 510,495" />
  </tableRegion>
  <tableRegion>
    <Coords points="50,400 510,400 50,495 510,495" />
  </tableRegion>
</document>
```
Notice that the coordinates of the points come in this order:
`x0, y0, x1, y0, x0, y1, x1, y1`

#### Take confidence with costants
I have prepared two "costants" files: `dataset_costants.py` and `inference_constants.py`.
The first contains all those costants that are useful to use to create dataset, the second to make
inference with the frozen graph. If you just want to run the project you should modify only those two files.
 
#### Transform the images from RGB to single-channel 8-bit grayscale jpeg images
Since colors are not useful for table detection, we can convert all the images in `.jpeg` 8-bit single channel images.
[This](https://www.researchgate.net/publication/320243569_Table_Detection_Using_Deep_Learning))
transformation is still under testing.
Use `python dataset/img_to_jpeg.py` after setting `dataset_costants.py`:
* `DPI_EXTRACTION`: output quality of the images;
* `PATH_TO_IMAGES`: path/to/datase/images;
* `IMAGES_EXTENSION`: extension of the extracted images. The only one tested is `.jpeg`.

#### Prepare the dataset for Tensorflow
The dataset was take from 
[TableBank](https://github.com/doc-analysis/TableBank).
It's a huge dataset apt to table detection (detect tables existance and borders)
and table recognition (understand table structure, heading, cells and so on).
It comes with a huge file of .json annotations: this has been transformed many .xml annotation files,
equal to the old annotations of this project.

Tensorflow instead can build its own TFRecord from csv informations, so we need to convert
the `xml` files into a `csv` one.
Use `python dataset/generate_database_csv.py` to do this conversion after setting `dataset_costants.py`:
* `TRAIN_CSV_NAME`: name for `.csv` train output file; 
* `TEST_CSV_NAME`: name for `.csv` test output file;
* `TRAIN_CSV_TO_PATH`: folder path for `TRAIN_CSV_NAME`;
* `TEST_CSV_TO_PATH`: folder path for `TEST_CSV_NAME`;
* `ANNOTATIONS_EXTENSION`: extension of annotations. In our case is `.xml`;
* `TRAINING_PERCENTAGE`: percentage of images for training
* `TEST_PERCENTAGE`: percentage of images for testing
* `TABLE_DICT`: dictionary for data labels. For this project there is no reason to change it;
* `MIN_WIDTH_BOX`, `MIN_HEIGHT_BOX`: minimum dimension to consider a box valid;
Some networks don't digest well little boxes, so I put this check.

#### Generate TF records file
`csv` files and images are ready: now we need to create our TF record file to feed Tensorflow.
Use `python generate_tf_records.py` to create the train and test`.record` files that we will need later. No need to configure
`dataset_costants.py`

#### Train the network
This part can be a little tricky. Let's divide it in steps.


#####Step 1: importing a pre-trained model
Look in [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
where there are pre-trained models for you to use.
All the info about how to configure and start a model can be found
[here](https://github.com/tensorflow/models/tree/master/research/object_detection)
First of all, choose a model, download it and save it in the folder `trained_nets`, where are your pre-trained models will be.

#####Step 2: create you model
Go to trained_models and create a new folder with the name you prefer.
Create two files inside it:
* `command.txt`: here you will write the commands useful to start the training and export the graph
* `myconfig.config`: this is a config file. You can find some examples
[here](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs).
Be careful to choose the correct config for your network.
There are a few that you must to set to set in the config. 
* `fine_tune_checkpoint`: this is the path to the `model.ckpt` file of the frozen graph.
* `input_path`x2: path to the record files `train_jpeg.record` and `test_jpeg.record` we made before
* `label_path`x2: path to the labels contained in the file `object-detection.pbtxt`.

Other parameters can be chosen. It's not very clear how to do it: you can find little info
[here](https://github.com/tensorflow/models/tree/master/research/object_detection/protos)
Once you've done your config, you can go to step 3.

#####Step 3: train your network
Write this command in the `command` file to reuse it. 
Example of the command to launch from `tensorflow/models/research/object-detection`
```angular2html
python model_main.py \
--pipeline_config_path="path/to/your/myconfig.config" \
--model_dir="path where you save your model, its nice if it is the same folder of myconfig.config" \
--num_train_steps=10000 \
--num_eval_steps=500 \
--alsologtostderr
```
Other options are inside `tensorflow/models/research/object-detection/model_main.py`
Notice: the paths must be your local paths
Notice: if you have already trained this network,
num_train_steps must be > of the steps the network has been trained already.

Monitor the training with [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard)
to check how it is going.

#### Export the graph
Use the command, from `tensorflow/models/research/object_detection`: 
```angular2html
python export_inference_graph.py \ 
--input_type=image_tensor \
--pipeline_config_path=path/to/automatically/created/pipeline.config \ 
--trained_checkpoint_prefix=path/to/last/model.ckpt-xxx \
--output_directory=path/to/output/dir
```

#### Test your graph!
Now that you have your graph you can try it out:
Run `inference_with_net.py` and set `inference_costants.py`:
* PATH_TO_GRAPH = os.path.abspath('trained_models/adam2_tb/frozen_adam2_tb/frozen_inference_graph.pb')
* TEST_SCORES: it's already set.
* NUM_CLASSES = the number of classes to detect. In our case it's 1. It's already set
* PATH_TO_LABELS = path to `.pbtxt` label file;
* MAX_NUM_BOXES = max number of boxes to be considered;
* PATH_TO_TEST_IMAGES_INPUT_FOLDER: take the input here to apply inference
* PATH_TO_TEST_IMAGES_OUTPUT_FOLDER the output images go here

The output will be a set of images with boxes drawn on it representing the inferred tables,
one image for each score.

In addition it will print a "merged" version of the boxes, in which
all the best vertically overlapping boxes are merged together to gain accuracy. `TEST_SCORES` is a list of
numbers that tells the program which scores must be merged together.

The procedure is better described in `inference_with_net.py`.
For every execution a `.log` file will be produced and put in `logs`.


## Common issues while installing Tensorflow models
### TypeError: can't pickle dict_values objects
[This](https://github.com/tensorflow/models/issues/4780#issuecomment-405441448)
comment will probably solve your problem.

### Windows build and python3 support for COCO API dataset
[This](https://github.com/philferriere/cocoapi)
clone will provide a working source for COCO API in Windows and Python3
