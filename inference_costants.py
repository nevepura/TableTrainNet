import os

PATH_TO_GRAPH = os.path.abspath('trained_models/adam2_tb/frozen_adam2_tb/frozen_inference_graph.pb')
TEST_SCORES = [0.2, 0.4, 0.6, 0.8]
NUM_CLASSES = 1
PATH_TO_LABELS = os.path.abspath('data/object-detection.pbtxt')
MAX_NUM_BOXES = 10
PATH_TO_TEST_IMAGES_INPUT_FOLDER = "test/input"
PATH_TO_TEST_IMAGES_OUTPUT_FOLDER = "test/output"
