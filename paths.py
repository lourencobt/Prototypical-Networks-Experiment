import os
import sys

# ! TO REMOVE WHEN IMPLEMENTATION IS FINISHED 
# * This environment variables are meant to be defined by the user of the meta_dataset_reader
os.environ['META_DATASET_ROOT'] = "/home/guests/lbt/meta-dataset"
os.environ['META_RECORDS_ROOT'] = "/home/guests/lbt/data/records"

# Read Environment Variables to define META_DATASET and RECORDS path
META_DATASET_ROOT = os.environ['META_DATASET_ROOT']
META_RECORDS_ROOT = os.environ['META_RECORDS_ROOT']

# Path of the meta_dataset_reader Gin configuration file 
GIN_CONFIG_ROOT = os.path.abspath('data/gin/meta_dataset_config.gin')

PROJECT_ROOT = '/'.join(os.path.realpath(__file__).split('/')[:-1])
META_DATA_ROOT = '/'.join(META_RECORDS_ROOT.split('/')[:-1])

