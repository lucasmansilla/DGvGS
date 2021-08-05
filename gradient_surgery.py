import torch
from random import shuffle


def get_agreement_func(method):
    if method == 'agr-sum':
        return agreement_sum
    elif method == 'agr-rand':
        return agreement_rand
    elif method == 'pcgrad':
        return pcgrad
    else:
        raise ValueError


def agreement_sum(domain_grads):
    """ Agr-Sum consensus strategy."""
    # Compute the agreement mask
    agr_mask = compute_agr_mask(domain_grads)

    # Sum the components that have the same sign and zero those that do not
    new_grads = torch.stack(domain_grads).sum(0)
    new_grads *= agr_mask

    return new_grads


def agreement_rand(domain_grads):
    """ Agr-Rand consensus strategy. """
    # Compute the agreement mask
    agr_mask = compute_agr_mask(domain_grads)

    # Sum the components that have the same sign
    new_grads = torch.stack(domain_grads).sum(0)
    new_grads *= agr_mask

    # Get a sample for the components that do not agree
    sample = torch.randn((~agr_mask).sum(), device=new_grads.device)
    scale = new_grads[agr_mask].abs().mean()
    # scale = new_grads.abs().mean()
    sample *= scale

    # Assign the values to these components
    new_grads[~agr_mask] = sample

    return new_grads


def pcgrad(domain_grads):
    """ Projecting conflicting gradients (PCGrad). """
    task_order = list(range(len(domain_grads)))

    # Run tasks in random order
    shuffle(task_order)

    # Initialize task gradients
    grad_pc = [g.clone() for g in domain_grads]

    for i in task_order:

        # Run other tasks
        other_tasks = [j for j in task_order if j != i]

        for j in other_tasks:
            grad_j = domain_grads[j]

            # Compute inner product and check for conflicting gradients
            inner_prod = torch.dot(grad_pc[i], grad_j)
            if inner_prod < 0:
                # Sustract the conflicting component
                grad_pc[i] -= inner_prod / (grad_j ** 2).sum() * grad_j

    # Sum task gradients
    new_grads = torch.stack(grad_pc).sum(0)

    return new_grads


def compute_agr_mask(domain_grads):
    """ Agreement mask. """
    grad_sign = torch.stack([torch.sign(g) for g in domain_grads])
    # True if all componentes agree, False if not
    agr_mask = torch.where(grad_sign.sum(0).abs() == len(domain_grads), 1, 0)
    return agr_mask.bool()
