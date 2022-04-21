import torch
import itertools as it
import numpy as np

from src.gradient import agreement_mask


def get_batch_same_class(dataset, class_label, batch_size, replace=True):
    """ Get a batch of data from one class. """
    class_indices = np.where(np.asarray(dataset.labels) == class_label)[0]
    batch_indices = np.random.choice(class_indices, batch_size, replace=replace)

    images = torch.stack([dataset[i][0] for i in batch_indices])
    target = torch.tensor([dataset[i][1] for i in batch_indices], device=images.device)

    return images, target


def cos_sim(domain_grads):
    """ Cosine similarity between domain gradients. """
    combinations = it.combinations(range(len(domain_grads)), 2)

    values = []
    for i, j in combinations:
        grad_i, grad_j = domain_grads[i], domain_grads[j]

        inner_prod = torch.dot(grad_i, grad_j)
        norm_prod = torch.norm(grad_i, 2) * torch.norm(grad_j, 2)
        cos_ij = inner_prod / norm_prod

        values.append(cos_ij.item())

    return np.mean(values)


def sign_sim(domain_grads):
    """ Sign similarity (%) between domain gradients. """
    agr_mask = agreement_mask(domain_grads) * 1.
    return agr_mask.mean().item()


def intra_cos_sim(domain_grads):
    """ Intra-domain cosine similarity. """
    values = [cos_sim(grads) for grads in domain_grads]
    return np.mean(values)


def inter_cos_sim(domain_grads):
    """ Inter-domain cosine similarity. """
    n_doms, n_examples = len(domain_grads), len(domain_grads[0])
    combinations = it.combinations(range(n_doms), 2)
    products = it.product(range(n_examples), range(n_examples))

    values = []
    for dom_i, dom_j in combinations:
        for sample_k, sample_l in products:
            grad_ik = domain_grads[dom_i][sample_k]
            grad_jl = domain_grads[dom_j][sample_l]
            values.append(cos_sim([grad_ik, grad_jl]))

    return np.mean(values)
