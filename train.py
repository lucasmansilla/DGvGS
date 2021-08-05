import argparse
import os
import random
import numpy as np
import torch

from models import get_model
import datasets
import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default='PACS')
    parser.add_argument('--method', type=str, default='deep-all')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--val_every', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--test_dom_idx', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default="result/train")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('Setup:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    print('Setting up model', end=' ', flush=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = vars(datasets)[args.dataset](args.data_dir, args.test_dom_idx)
    model = get_model(device, dataset, args)
    print('Ok')

    print('Training')
    model.train()

    print('Testing')
    model.test()

    train_stats = model.get_train_stats()
    test_acc, test_loss = train_stats['acc']['test'], train_stats['loss']['test']
    print(f'\ttest loss: {test_loss:.5f}, test acc: {test_acc:.2f}%')

    print('Saving training stats', end=' ', flush=True)
    utils.save_train_stats(train_stats, args.output_dir + '/train_stats.pkl')
    with open(args.output_dir + '/train_info.txt', 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write('{}: {}\n'.format(k, v))
        f.write('test_loss: {:.5f}\n'.format(test_loss))
        f.write('test_acc: {:.2f}\n'.format(test_acc))
    print('Ok')
