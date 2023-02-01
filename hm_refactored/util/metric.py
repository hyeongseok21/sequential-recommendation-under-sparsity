import numpy as np
from scipy.sparse import csr_matrix, triu


def hit(gt_item, pred_item, batch=False):
    result = []
    if batch:
        for gt, pred in zip(gt_item, pred_item):
            if len(gt) == 0:
                continue
            tmp = 0
            for p in pred:
                if p in gt:
                    tmp = 1
                    break
            result.append(tmp)
    else:
        tmp = 0
        for pred in pred_item:
            if pred in gt_item:
                tmp = 1
                break
        result.append(tmp)
    return result
        
def ndcg(gt_item, pred_item, batch=False):
    result = []
    if batch:
        for gt, pred in zip(gt_item, pred_item):
            if len(gt) == 0:
                continue
            tmp = [0]
            for g in set(gt):
                if g in pred:
                    index = pred.index(g)
                    tmp.append(np.reciprocal(np.log2(index+2)))
            result.append(max(tmp))
    else:
        tmp = [0]
        for gt in set(gt_item):
            if gt in pred_item:
                index = pred_item.index(gt)
                tmp.append(np.reciprocal(np.log2(index+2)))
        result.append(max(tmp))
    return result

def map_(gt_items, pred_items, k, batch=False):
    result = []
    if batch:
        for gt, pred in zip(gt_items, pred_items):
            if len(gt) == 0:
                continue
            score = 0.0
            num_hits = 0.0
            for i, p in enumerate(pred):
                if p in gt and p not in pred[:i]:
                    num_hits += 1.0
                    score += num_hits / (i+1.0)
            result.append(score / min(len(gt), k))
        return result
    else:
        if len(gt_items) == 0:
            return result
        score = 0.0
        num_hits = 0.0
        for i, p in enumerate(pred_items):
            if p in gt_items and p not in pred_items[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)
        result.append(score / min(len(gt_items), k))
    return result

def diversity(user_result, num_user, num_item, topk=10):
    row = []
    col = []
    data = []
    for u in user_result:
        if user_result[u] == {} or user_result[u]['user_item_count'] == 0:
            continue
        row += ([u] * topk)
        col += user_result[u]['rec']
        data += ([1] * topk)
    m = csr_matrix((data, (row, col)), shape=(num_user, num_item))
    co_matrix = m * m.T
    co_matrix = triu(co_matrix, k=1, format='csr')
    co_matrix = co_matrix / topk
    return 1 - (co_matrix.sum() / len(co_matrix.nonzero()[0]))