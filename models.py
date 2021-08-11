import torch

from networks import AlexNet
from dataloader import InfiniteDataLoader
from gradient_surgery import get_agreement_func


def get_model(device, dataset, args):
    if args.method == 'deep-all':
        return ModelDA(device, dataset, args)
    elif args.method in ['agr-sum', 'agr-rand', 'pcgrad']:
        return ModelGS(device, dataset, args)
    else:
        raise ValueError


class ModelDA:
    """ Baseline model (Deep-All). """

    def __init__(self, device, dataset, args):
        self.device = device
        self.args = args
        self.network = AlexNet(dataset.N_CLASSES).to(device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay)
        self._create_dataloaders(dataset, args)
        self.stats = {'loss': {'train': [], 'val': [], 'test': []},
                      'acc':  {'train': [], 'val': [], 'test': []}}

    def _create_dataloaders(self, dataset, args):

        def get_dataloader(dataset, batch_size, is_train=False):
            if is_train:
                return InfiniteDataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True)
            else:
                return torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False)

        self.train_loaders = []
        for dom_dataset in dataset['train']:
            self.train_loaders.append(get_dataloader(dom_dataset, args.batch_size, True))
        self.val_loader = get_dataloader(dataset['val'], args.batch_size)
        self.test_loader = get_dataloader(dataset['test'], args.batch_size)

    def _prepare_batch(self, batch):
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        return inputs, targets

    def train(self):
        train_iterator = zip(*self.train_loaders)
        iterations = self.args.iterations
        val_every = self.args.val_every
        run_train_loss, run_train_acc = 0.0, 0.0
        max_val_acc = -1

        for it in range(iterations):
            # Training
            train_batches = [self._prepare_batch(batch) for batch in next(train_iterator)]
            train_loss, train_acc = self._train_step(train_batches)
            run_train_loss += train_loss
            run_train_acc += train_acc

            if it == 0 or (it+1) % val_every == 0 or it == (iterations-1):
                # Validation
                val_loss, val_acc = self._validation_step()

                # Save model when the validation accuracy increases
                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                    torch.save(self.network.state_dict(), self.args.output_dir + '/best_model.pt')

                if (it+1) % val_every == 0:
                    run_train_loss /= val_every
                    run_train_acc /= val_every
                elif it == (iterations-1):
                    n_steps = iterations
                    n_steps -= n_steps//val_every * val_every
                    run_train_loss /= n_steps
                    run_train_acc /= n_steps

                self.stats['loss']['train'].append(run_train_loss)
                self.stats['acc']['train'].append(run_train_acc)
                self.stats['loss']['val'].append(val_loss)
                self.stats['acc']['val'].append(val_acc)

                # Print stats
                print(f'\titer {it+1:>5}/{iterations}: '
                      f'train loss: {run_train_loss:.5f}, '
                      f'train acc: {run_train_acc:>6.2f}% | '
                      f'val loss: {val_loss:.5f}, '
                      f'val acc: {val_acc:>6.2f}%')

                run_train_loss, run_train_acc = 0.0, 0.0

    def _train_step(self, train_batches):
        is_train = True
        inputs = torch.cat([x for x, _ in train_batches])
        targets = torch.cat([y for _, y in train_batches])

        self.network.train(is_train)
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = self.network(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()

            predictions = torch.max(outputs, 1)[1]
            train_loss = loss.item()
            train_acc = (predictions == targets).float().mean().item() * 100

        self.optimizer.step()

        return train_loss, train_acc

    def _validation_step(self):
        is_train = False
        val_loss, val_acc = 0.0, 0.0

        self.network.train(is_train)

        with torch.set_grad_enabled(is_train):
            for batch in self.val_loader:
                inputs, targets = self._prepare_batch(batch)

                outputs = self.network(inputs)
                loss = self.loss_fn(outputs, targets)

                predictions = torch.max(outputs, 1)[1]
                val_loss += loss.item() * inputs.size(0)
                val_acc += (predictions == targets).sum().item()

        val_loss /= len(self.val_loader.dataset)
        val_acc /= len(self.val_loader.dataset) / 100

        return val_loss, val_acc

    def test(self):
        is_train = False
        test_loss, test_acc = 0.0, 0.0
        self.network.train(is_train)
        self.network.load_state_dict(torch.load(self.args.output_dir + '/best_model.pt'))

        with torch.set_grad_enabled(is_train):
            for batch in self.test_loader:
                inputs, targets = self._prepare_batch(batch)

                outputs = self.network(inputs)
                loss = self.loss_fn(outputs, targets)

                predictions = torch.max(outputs, 1)[1]
                test_loss += loss.item() * inputs.size(0)
                test_acc += (predictions == targets).sum().item()

        self.stats['loss']['test'] = test_loss / len(self.test_loader.dataset)
        self.stats['acc']['test'] = test_acc / len(self.test_loader.dataset) * 100

    def get_train_stats(self):
        return self.stats


class ModelGS(ModelDA):
    """ Model with gradient surgery. """

    def __init__(self, device, dataset, args):
        super().__init__(device, dataset, args)
        self.grad_fn = get_agreement_func(args.method)

    def _train_step(self, train_batches):
        is_train = True
        train_loss, train_acc = 0.0, 0.0
        domain_grads = []

        self.network.train()
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            for batch in train_batches:
                inputs, targets = batch

                outputs = self.network(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()

                domain_grads.append(self.network.get_grads())

                predictions = torch.max(outputs, 1)[1]
                train_loss += loss.item()
                train_acc += (predictions == targets).float().mean().item()

                self.optimizer.zero_grad()

        train_loss /= len(train_batches)
        train_acc /= len(train_batches)

        new_grads = self.grad_fn(domain_grads)  # Modify gradients according to grad_fn
        self.network.set_grads(new_grads)       # Update gradients
        self.optimizer.step()                   # Update model parameters

        return train_loss, train_acc
