import time
import math
import torch
import numpy as np

from utils.functions import load_model, move_to


def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]


def evaluate(model_path, dataset_path, num_nodes, temperature=1.0, prize_type='const', use_cuda=True):
    
    # Set device
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Load model
    model, _ = load_model(model_path)
    model.to(device)
    model.eval()
    model.set_decode_type("greedy", temp=temperature)

    # Load data
    dataset = model.problem.make_dataset(filename=dataset_path, size=num_nodes, num_samples=1, distribution=prize_type)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # Iterate
    results = []
    for batch in dataloader:
        batch = move_to(batch, device)

        start = time.time()
        with torch.no_grad():
            # This returns (batch_size, iter_rep shape)
            sequences, costs = model.sample_many(batch, batch_rep=1, iter_rep=1)
            batch_size = len(costs)
            ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)

        if sequences is None:
            sequences = [None] * batch_size
            costs = [math.inf] * batch_size
        else:
            sequences, costs = get_best(
                sequences.cpu().numpy(), costs.cpu().numpy(),
                ids.cpu().numpy() if ids is not None else None,
                batch_size
            )
        duration = time.time() - start
        for seq, cost in zip(sequences, costs):
            if model.problem.NAME == "tsp":
                seq = seq.tolist()  # No need to trim as all are same length
            elif model.problem.NAME in ("cvrp", "sdvrp"):
                seq = np.trim_zeros(seq).tolist() + [0]  # Add depot
            elif model.problem.NAME in ("op", "pctsp"):
                seq = np.trim_zeros(seq)  # We have the convention to exclude the depot
            else:
                assert False, "Unkown problem: {}".format(model.problem.NAME)
            # Note VRP only
            results.append((cost, seq, duration))

    return results[0]
