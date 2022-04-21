import argparse
import os
import subprocess

from src.datasets import get_domains
from src.utils.misc import get_seed_hash

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--num_trials', type=int, default=20)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--val_every', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--results_dir', type=str)
    args = parser.parse_args()

    results_dir = os.path.join(args.results_dir, args.dataset, args.method)
    os.makedirs(results_dir, exist_ok=True)

    domain_names = get_domains(args.dataset)

    for i in range(args.num_trials):
        print(f'\nTrial #{i+1}\n')

        for j, name in enumerate(domain_names):
            # Prepare arguments
            cmd_args = dict(vars(args))
            cmd_args.update({
                'seed': get_seed_hash(args.dataset, args.method, j, i),
                'test_domain_index': j,
                'results_dir': os.path.join(results_dir, f'{i+1}/{name}')
            })
            del cmd_args['num_trials']

            cmd = ['python', 'scripts/train_model.py ']
            for k, v in sorted(cmd_args.items()):
                if isinstance(v, str):
                    v = f'\'{v}\''
                cmd.append(f'--{k}={v}')
            cmd_str = ' '.join(cmd)

            # Run script
            subprocess.run(cmd_str, shell=True, check=True)

    print('\nDone.\n')
