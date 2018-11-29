'''
 @Date  : 7/23/2018
 @Author: Shuming Ma
 @mail  : shumingma@pku.edu.cn 
 @homepage: shumingma.com
'''

def calc_hit_rank(prediction, reference):
    for i, p in enumerate(prediction):
        if reference[p] == 1:
            return i+1
    print(prediction)
    print(reference)
    raise ValueError('No reference!')

def recall(predictions, references, k=1):
    assert len(predictions) == len(references)
    total = len(references)
    hits = 0
    for p, c in zip(predictions, references):
        hits += int(calc_hit_rank(p, c) <= k)
    return hits * 100.0 / total

def mean_rank(predictions, references):
    assert len(predictions) == len(references)
    ranks = []
    for p, c in zip(predictions, references):
        rank = calc_hit_rank(p, c)
        ranks.append(rank)
    return sum(ranks) * 1.0 / len(ranks)

def mean_reciprocal_rank(predictions, references):
    assert len(predictions) == len(references)
    ranks = []
    for p, c in zip(predictions, references):
        rank = calc_hit_rank(p, c)
        ranks.append(1.0 / rank)
    return sum(ranks) * 1.0 / len(ranks)

