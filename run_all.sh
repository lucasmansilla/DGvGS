#!/bin/bash

python train_all.py --data_dir=data/pre --output_dir=result/train_all --dataset=PACS --method=deep-all
python train_all.py --data_dir=data/pre --output_dir=result/train_all --dataset=PACS --method=agr-sum
python train_all.py --data_dir=data/pre --output_dir=result/train_all --dataset=PACS --method=agr-rand
python train_all.py --data_dir=data/pre --output_dir=result/train_all --dataset=PACS --method=pcgrad

python train_all.py --data_dir=data/pre --output_dir=result/train_all --dataset=VLCS --method=deep-all
python train_all.py --data_dir=data/pre --output_dir=result/train_all --dataset=VLCS --method=agr-sum
python train_all.py --data_dir=data/pre --output_dir=result/train_all --dataset=VLCS --method=agr-rand
python train_all.py --data_dir=data/pre --output_dir=result/train_all --dataset=VLCS --method=pcgrad

python train_all.py --data_dir=data/pre --output_dir=result/train_all --dataset=OfficeHome --method=deep-all
python train_all.py --data_dir=data/pre --output_dir=result/train_all --dataset=OfficeHome --method=agr-sum
python train_all.py --data_dir=data/pre --output_dir=result/train_all --dataset=OfficeHome --method=agr-rand
python train_all.py --data_dir=data/pre --output_dir=result/train_all --dataset=OfficeHome --method=pcgrad