import os
import random
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--test_split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    domain_names = sorted([f.name for f in os.scandir(args.data_dir) if f.is_dir()])
    class_names = sorted([f.name for f in os.scandir(os.path.join(args.data_dir, domain_names[0])) if f.is_dir()])

    print('\nSplitting domain data:\n')
    for domain_name in domain_names:
        print(f'\tDomain {domain_name}', end=' ', flush=True)

        # Get filenames
        data, y = [], []
        for i, class_name in enumerate(class_names):
            dir_path = os.path.join(args.data_dir, domain_name, class_name)
            files = sorted([f for f in os.listdir(dir_path)])
            data += [f'{domain_name}/{class_name}/{name},{i+1}' for name in files]
            y += [i + 1 for _ in range(len(files))]

        # Split into training, validation and test
        x_train, x_test, y_train, _ = train_test_split(data, y, test_size=args.test_split, stratify=y)
        x_train, x_val = train_test_split(x_train, test_size=args.val_split / (1 - args.test_split), stratify=y_train)

        # Save files
        np.savetxt(f'{args.output_dir}/{domain_name}_train.txt', sorted(x_train), fmt='%s')
        np.savetxt(f'{args.output_dir}/{domain_name}_val.txt', sorted(x_val), fmt='%s')
        np.savetxt(f'{args.output_dir}/{domain_name}_test.txt', sorted(x_test), fmt='%s')

        print('Ok')

print('\nDone.\n')
