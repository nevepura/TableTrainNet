import os
"""
All costants that are used to build the dataset are here.
PATH_TO_{} are those path in which the programs read
{}_TO_PATH are those in which the programs write
"""
DPI_EXTRACTION = 200

PATH_TO_IMAGES = os.path.dirname(os.path.abspath("dataset/Images"))

PATH_TO_ANNOTATIONS = os.path.dirname(os.path.abspath("dataset/Annotations"))

TRAIN_CSV_NAME = 'train_jpeg.csv'
TEST_CSV_NAME = 'test_jpeg.csv'

TRAIN_CSV_TO_PATH = os.path.dirname(os.path.abspath("data"))

TEST_CSV_TO_PATH = os.path.dirname(os.path.abspath("data"))

TF_TRAIN_RECORD_TO_PATH = os.path.dirname(os.path.abspath("data"))

TF_TRAIN_RECORD_NAME = 'train_jpeg.record'

TF_TEST_RECORD_TO_PATH = os.path.dirname(os.path.abspath("data"))
TF_TEST_RECORD_NAME = 'test_jpeg.record'
ANNOTATIONS_EXTENSION = '.xml'
IMAGES_EXTENSION = '.jpeg'

TRAINING_PERCENTAGE = 0.995
TEST_PERCENTAGE = 0.005

MIN_WIDTH_BOX = 1
MIN_HEIGHT_BOX = 1

TABLE_DICT = {
	'id': '1',
	'name': 'table'
}