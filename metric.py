import numpy as np


def hit(rank, ground_truth):
    # HR is equal to Recall when dataset is loo split.
    count = 0
    # 点击率
    for idx, item in enumerate(rank):
        if item in ground_truth:
            count += 1
            break
    return count


def precision(rank, ground_truth):
    # Precision is meaningless when dataset is loo split.
    hits = [1 if item in ground_truth else 0 for item in rank]
    result = np.cumsum(hits, dtype=np.float32) / np.arange(1, len(rank) + 1)
    return result


def recall(rank, ground_truth):
    # Recall is equal to HR when dataset is loo split.
    hits = [1 if item in ground_truth else 0 for item in rank]
    result = np.cumsum(hits, dtype=np.float32) / len(ground_truth)
    return result


def map(rank, ground_truth):
    # Reference: https://blog.csdn.net/u010138758/article/details/69936041
    # MAP is equal to MRR when dataset is loo split.
    pre = precision(rank, ground_truth)
    pre = [pre[idx] if item in ground_truth else 0 for idx, item in enumerate(rank)]
    sum_pre = np.cumsum(pre, dtype=np.float32)
    relevant_num = np.cumsum([1 if item in ground_truth else 0 for item in rank])
    result = [p / r_num if r_num != 0 else 0 for p, r_num in zip(sum_pre, relevant_num)]
    return result


def ndcg(rank, ground_truth):
    len_rank = len(rank)
    idcg_len = min(len(ground_truth), len_rank)
    idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
    idcg[idcg_len:] = idcg[idcg_len - 1]
    dcg = np.cumsum([1.0 / np.log2(idx + 2) if item in ground_truth
                     else 0.0 for idx, item in enumerate(rank)])
    result = dcg / idcg
    return result[-1]


def mrr(rank, ground_truth):
    # MRR is equal to MAP when dataset is loo split.
    last_idx = 0
    for idx, item in enumerate(rank, start=1):
        if item in ground_truth:
            last_idx = idx
            break
    if last_idx == 0:
        return 0
    else:
        return 1 / last_idx
