import torch

from src.datasets import get_classes
from src.networks import AlexNet
from src.gradient import METHODS, get_method


def get_model(args):
    if args.method == 'deep-all':
        return DeepAll(args)
    elif args.method in METHODS.keys():
        return GradSurgery(args)
    else:
        raise ValueError


class DeepAll:

    def __init__(self, args):
        self.device = 'cuda'

        self.network = AlexNet(num_classes=get_classes(args.dataset), use_pretrained=True)
        self.network = torch.nn.DataParallel(self.network)
        self.network.to(self.device)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.network.module.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    def train(self, train_batches):
        images = torch.cat([x for x, _ in train_batches]).to(self.device)
        target = torch.cat([y for _, y in train_batches]).to(self.device)

        # Training mode
        self.network.module.train()

        output = self.network(images)
        loss = self.loss_fn(output, target)

        train_loss = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return train_loss

    def validate(self, val_loader):

        # Validation mode
        self.network.module.eval()

        with torch.no_grad():

            val_loss = 0.0
            val_acc = 0.0
            num_val_examples = 0

            for images, target in val_loader:
                images, target = images.to(self.device), target.to(self.device)

                output = self.network(images)
                loss = self.loss_fn(output, target)
                predicted = torch.max(output, 1)[1]

                num_val_examples += images.size(0)
                val_loss += loss.item() * images.size(0)
                val_acc += (predicted == target).sum().item()

        val_loss /= num_val_examples
        val_acc /= num_val_examples

        return val_loss, val_acc

    def load(self, path):
        self.network.module.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.network.module.state_dict(), path)


class GradSurgery(DeepAll):

    def __init__(self, args):
        super().__init__(args)
        self.grad_fn = get_method(args.method)

    def train(self, train_batches):

        # Training mode
        self.network.module.train()

        domain_grads = []
        train_loss = 0.0

        for images, target in train_batches:
            images, target = images.to(self.device), target.to(self.device)

            output = self.network(images)
            loss = self.loss_fn(output, target)

            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            domain_grads.append(self.network.module.get_grads())

        train_loss /= len(train_batches)

        new_grads = self.grad_fn(domain_grads)    # modify gradients according to grad_fn
        self.network.module.set_grads(new_grads)  # update gradients
        self.optimizer.step()                     # update model parameters

        return train_loss
