import os
import random
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/pre/PACS/images')
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--output_dir', type=str, default='data/pre/PACS/split')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    domains = sorted([f.name for f in os.scandir(args.data_dir) if f.is_dir()])
    classes = sorted([f.name for f in os.scandir(os.path.join(args.data_dir,
                      domains[0])) if f.is_dir()])

    for dom_name in domains:

        print(f'Processing {dom_name}', end=' ', flush=True)

        # Get filenames
        dom_data, y = [], []
        for i, class_name in enumerate(classes):
            dir_path = os.path.join(args.data_dir, dom_name, class_name)
            files = sorted([f for f in os.listdir(dir_path)])
            dom_data += [f'{dom_name}/{class_name}/{fname},{i+1}'
                         for fname in files]
            y += [i+1 for _ in range(len(files))]

        # Split into training, validation and testing
        train, test, y_train, _ = train_test_split(
            dom_data, y, test_size=args.test_size, stratify=y)
        train, val = train_test_split(
            train, test_size=args.val_size/(1-args.test_size), stratify=y_train)

        # Save files
        np.savetxt(f'{args.output_dir}/{dom_name}_train.txt', sorted(train), fmt='%s')
        np.savetxt(f'{args.output_dir}/{dom_name}_val.txt', sorted(val), fmt='%s')
        np.savetxt(f'{args.output_dir}/{dom_name}_test.txt', sorted(test), fmt='%s')

        print('Ok')
