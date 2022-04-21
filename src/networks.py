import torch
import torchvision.models


class AlexNet(torch.nn.Module):

    def __init__(self, num_classes, use_pretrained=False):
        super().__init__()
        self.network = torchvision.models.alexnet(pretrained=use_pretrained)
        self.network.classifier[6] = torch.nn.Linear(4096, num_classes)

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

    def set_grads(self, new_grads):
        start = 0
        for k, p in enumerate(self.network.parameters()):
            dims = p.shape
            end = start + dims.numel()
            p.grad.data = new_grads[start:end].reshape(dims)
            start = end
