import argparse
import os
import random
import time
import torch
import numpy as np

from src.datasets import get_dataset
from src.dataloader import InfiniteDataLoader
from src.models import get_model
from src.utils.io import save_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--val_every', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--test_domain_index', type=int, default=0)
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--save_model', action='store_true')
    args = parser.parse_args()

    print('\nArgs:\n')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.results_dir, exist_ok=True)

    # Configure device
    num_gpus = len(args.gpu.split(','))
    assert num_gpus == 1 or (num_gpus > 1 and args.batch_size % num_gpus == 0)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Configure model and dataset
    model = get_model(args)
    dataset = get_dataset(args)

    def create_dataloader(dataset, batch_size, is_train=True):
        if is_train:
            return InfiniteDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
            )
        else:
            return torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False
            )

    # Create dataloaders
    train_loader = [create_dataloader(d, args.batch_size) for d in dataset['train']]
    val_loader = create_dataloader(dataset['val'], args.batch_size, False)
    test_loader = create_dataloader(dataset['test'], args.batch_size, False)

    train_iterator = zip(*train_loader)

    train_results = {
        'train_loss': [],
        'train_time': [],
        'val_loss': [],
        'val_acc': []
    }

    train_loss_list = []
    train_time_list = []
    best_val_acc = -1

    print('\nTraining:\n')
    for it in range(args.iterations):
        t_start = time.time()

        # Training
        train_batches = [batch for batch in next(train_iterator)]
        loss = model.train(train_batches)

        train_loss_list.append(loss)
        train_time_list.append(time.time() - t_start)

        if (it % args.val_every == 0) or (it == args.iterations - 1):

            train_loss = np.mean(train_loss_list)
            train_results['train_loss'].append(train_loss)

            train_time = np.mean(train_time_list)
            train_results['train_time'].append(train_time)

            # Validation
            val_loss, val_acc = model.validate(val_loader)

            train_results['val_loss'].append(val_loss)
            train_results['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

                # Save model when validation accuracy increases
                if args.save_model:
                    model.save(os.path.join(args.results_dir, 'model_best.pt'))

            print(f'\t{it:>5}/{args.iterations-1} '
                  f'train_loss: {train_loss:.4f} '
                  f'val_loss: {val_loss:.4f} '
                  f'val_acc: {val_acc:.4f} '
                  f'step_time: {train_time:.2f} sec')

            train_loss_list = []
            train_time_list = []

    # Testing
    test_loss, test_acc = model.validate(test_loader)

    train_results.update({'test_loss': test_loss, 'test_acc': test_acc})
    save_dict(train_results, os.path.join(args.results_dir, 'train_results.pkl'))

    with open(os.path.join(args.results_dir, 'run_args.txt'), 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write(f'{k}: {v}\n')

    print('\nDone.\n')
