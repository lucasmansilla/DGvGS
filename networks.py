import torch
import torchvision.models


class AlexNet(torch.nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.network = torchvision.models.alexnet(pretrained=True)
        self.network.classifier[6] = torch.nn.Linear(4096, n_classes)

    def forward(self, x):
        return self.network(x)

    def get_weights(self):
        weights = []
        for p in self.network.parameters():
            weights.append(p.data.clone().flatten())
        return torch.cat(weights)

    def get_grads(self):
        grads = []
        for p in self.network.parameters():
            grads.append(p.grad.data.clone().flatten())
        return torch.cat(grads)

    def update_grads(self, new_grads):
        start = 0
        for k, p in enumerate(self.network.parameters()):
            dims = p.shape
            end = start + dims.numel()
            p.grad.data = new_grads[start:end].reshape(dims)
            start = end
