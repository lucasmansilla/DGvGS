import argparse
import os
import subprocess

import datasets
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--dataset', type=str, default='PACS')
    parser.add_argument('--method', type=str, default='deep-all')
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--val_every', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--output_dir', type=str, default='result/train_all/')
    parser.add_argument('--n_trials', type=int, default=20)
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.dataset, args.method)
    os.makedirs(output_dir, exist_ok=True)

    domains = datasets.get_domains(args.dataset)
    n_test_doms = len(domains)

    for trial_i in range(args.n_trials):
        for test_idx in range(n_test_doms):
            train_args = {}
            train_args['data_dir'] = args.data_dir
            train_args['dataset'] = args.dataset
            train_args['method'] = args.method
            train_args['seed'] = utils.seed_generator(args.dataset, args.method, test_idx, trial_i)
            train_args['iterations'] = args.iterations
            train_args['val_every'] = args.val_every
            train_args['batch_size'] = args.batch_size
            train_args['lr'] = args.lr
            train_args['weight_decay'] = args.weight_decay
            train_args['test_dom_idx'] = test_idx
            train_args['output_dir'] = os.path.join(
                output_dir, f'run_{trial_i}/{domains[test_idx]}')

            command = ['python', 'train.py ']
            for k, v in sorted(train_args.items()):
                if isinstance(v, str):
                    v = f'\'{v}\''
                command.append(f'--{k}={v}')
            command_str = ' '.join(command)

            subprocess.run(command_str, shell=True, check=True)
