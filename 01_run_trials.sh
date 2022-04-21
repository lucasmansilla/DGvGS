#!/bin/bash

# PACS
python scripts/run_trials.py --data_dir=data/processed --results_dir=results/trials --dataset=PACS --method=deep-all
python scripts/run_trials.py --data_dir=data/processed --results_dir=results/trials --dataset=PACS --method=agr-sum
python scripts/run_trials.py --data_dir=data/processed --results_dir=results/trials --dataset=PACS --method=agr-rand
python scripts/run_trials.py --data_dir=data/processed --results_dir=results/trials --dataset=PACS --method=pcgrad

# VLCS
python scripts/run_trials.py --data_dir=data/processed --results_dir=results/trials --dataset=VLCS --method=deep-all
python scripts/run_trials.py --data_dir=data/processed --results_dir=results/trials --dataset=VLCS --method=agr-sum
python scripts/run_trials.py --data_dir=data/processed --results_dir=results/trials --dataset=VLCS --method=agr-rand
python scripts/run_trials.py --data_dir=data/processed --results_dir=results/trials --dataset=VLCS --method=pcgrad

# Office-Home
python scripts/run_trials.py --data_dir=data/processed --results_dir=results/trials --dataset=OfficeHome --method=deep-all
python scripts/run_trials.py --data_dir=data/processed --results_dir=results/trials --dataset=OfficeHome --method=agr-sum
python scripts/run_trials.py --data_dir=data/processed --results_dir=results/trials --dataset=OfficeHome --method=agr-rand
python scripts/run_trials.py --data_dir=data/processed --results_dir=results/trials --dataset=OfficeHome --method=pcgrad