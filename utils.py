import hashlib
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools as it
from gradient_surgery import compute_agr_mask


def save_train_stats(stats, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(stats, f)


def load_train_stats(file_path):
    with open(file_path, 'rb') as f:
        stats = pickle.load(f)
    return stats


def plot_train_stats(stats, path):
    plt.figure(figsize=(9.6, 4.8))

    plt.subplot(1, 2, 1)
    plt.plot(stats['loss']['train'], 'b', label='Training')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(stats['acc']['val'], 'r', label='Validation')
    plt.ylim([0.0, 100.0])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation accuracy')
    plt.legend()

    plt.savefig(path)
    plt.close()


def get_batch_class(dataset, class_label, batch_size, replace=True):
    class_indices = np.where(np.asarray(dataset.labels) == class_label)[0]
    batch_indices = np.random.choice(class_indices, batch_size, replace=replace)
    inputs = torch.stack([dataset[i][0] for i in batch_indices])
    targets = torch.tensor([dataset[i][1] for i in batch_indices], device=inputs.device)
    return inputs, targets


def seed_generator(*args):
    m = hashlib.md5(str(args).encode('utf-8'))
    h = m.hexdigest()
    i = int(h, 16)
    seed = i % (2**31)
    return seed


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
    agr_mask = compute_agr_mask(domain_grads) * 1.
    return agr_mask.mean().item()


def intra_cos_sim(domain_grads):
    """ Intra-domain cosine similarity. """
    values = [cos_sim(grads) for grads in domain_grads]
    return np.mean(values)


def inter_cos_sim(domain_grads):
    """ Inter-domain cosine similarity. """
    n_doms, n_samples = len(domain_grads), len(domain_grads[0])
    combinations = it.combinations(range(n_doms), 2)
    products = it.product(range(n_samples), range(n_samples))
    values = []
    for dom_i, dom_j in combinations:
        for sample_k, sample_l in products:
            grad_ik = domain_grads[dom_i][sample_k]
            grad_jl = domain_grads[dom_j][sample_l]
            values.append(cos_sim([grad_ik, grad_jl]))
    return np.mean(values)
