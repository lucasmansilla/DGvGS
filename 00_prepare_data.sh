#!\bin\bash

# PACS
python scripts/resize_images.py --data_dir=data/raw/PACS/kfold --output_dir=data/processed/PACS/images
python scripts/split_data.py --data_dir=data/processed/PACS/images --output_dir=data/processed/PACS/split

# VLCS
python scripts/resize_images.py --data_dir=data/raw/VLCS --output_dir=data/processed/VLCS/images
python scripts/split_data.py --data_dir=data/processed/VLCS/images --output_dir=data/processed/VLCS/split

# Office-Home
python scripts/resize_images.py --data_dir=data/raw/OfficeHomeDataset_10072016 --output_dir=data/processed/OfficeHome/images
python scripts/split_data.py --data_dir=data/processed/OfficeHome/images --output_dir=data/processed/OfficeHome/split

