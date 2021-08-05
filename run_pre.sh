#!/bin/bash

python -m preprocess.resize --data_dir=data/orig/PACS/kfold --output_dir=data/pre/PACS/images
python -m preprocess.resize --data_dir=data/orig/VLCS --output_dir=data/pre/VLCS/images
python -m preprocess.resize --data_dir=data/orig/OfficeHomeDataset_10072016 --output_dir=data/pre/OfficeHome/images

python -m preprocess.split --data_dir=data/pre/PACS/images --output_dir=data/pre/PACS/split
python -m preprocess.split --data_dir=data/pre/VLCS/images --output_dir=data/pre/VLCS/split
python -m preprocess.split --data_dir=data/pre/OfficeHome/images --output_dir=data/pre/OfficeHome/split