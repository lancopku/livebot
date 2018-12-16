'''
 @Date  : 8/15/2018
 @Author: Shuming Ma
 @mail  : shumingma@pku.edu.cn 
 @homepage: shumingma.com
'''
import json
import time
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer

def load_from_json(fin):
    datas = []
    for line in fin:
        data = json.loads(line)
        datas.append(data)
    return datas

def dump_to_json(datas, fout):
    for data in datas:
        fout.write(json.dumps(data, sort_keys=True, separators=(',', ': '), ensure_ascii=False))
        fout.write('\n')
    fout.close()

def get_comment_set(comment_list):
    comment_set = {}
    for comment in comment_list:
        if comment in comment_set:
            comment_set[comment] += 1
        else:
            comment_set[comment] = 1
    comment_set = sorted(comment_set.items(), key=lambda kv: -kv[1])
    comment_set = list(zip(*comment_set))[0]
    return comment_set

def get_correct_set(news, candidate_set):
    for comment in news['comment']:
        if comment not in candidate_set:
            candidate_set[comment] = 1
    return candidate_set

def get_popular_set(comment_set, candidate_set, k):
    for comment in comment_set[:k]:
        if comment not in candidate_set:
            candidate_set[comment] = 1
    return candidate_set

def get_random_set(comment_set, candidate_set, k):
    while len(candidate_set) < k:
        rand = random.randint(0, len(comment_set) - 1)
        if comment_set[rand] not in candidate_set:
            candidate_set[comment_set[rand]] = 1
    return candidate_set

def get_plausible_set(comment_set, candidate_set, k, query_tfidf, comment_tfidf):
    matrix = (query_tfidf * comment_tfidf.transpose()).todense()
    ids = np.array(np.argsort(-matrix, axis=1))[0]
    for id in ids[:k]:
        if comment_set[id] not in candidate_set:
            candidate_set[comment_set[id]] = 1
    return candidate_set

def get_candidate_set(fin, fout, comment_list, tvec, comment_tfidf):
    datas = load_from_json(fin)
    newdatas = []
    for data in datas:
        candidate_set = {}
        candidate_set = get_correct_set(data, candidate_set)
        candidate_set = get_popular_set(comment_list, candidate_set, 20)
        candidate_set = get_plausible_set(comment_list, candidate_set, 20, tvec.transform([data['context']]), comment_tfidf)
        candidate_set = get_random_set(comment_list, candidate_set, 100)
        newdatas.append({'video': data['video'], 'time': data['time'],
                         'context': data['context'], 'comment': data['comment'],
                         'candidate': candidate_set})
    dump_to_json(newdatas, fout)


if __name__ == '__main__':
    comments = []
    for data in load_from_json(open('train.json', 'r', encoding='utf8')):
        comments.append(data['comment'])

    comment_set = get_comment_set(comments)
    tvec = TfidfVectorizer()
    tvec = tvec.fit(comment_set)
    comment_tfidf = tvec.transform(comment_set)

    get_candidate_set(open('test-context.json', 'r', encoding='utf8'), open('test-candidate.json', 'w', encoding='utf8'),
                      comment_set, tvec, comment_tfidf)

    get_candidate_set(open('dev-context.json', 'r', encoding='utf8'), open('dev-candidate.json', 'w', encoding='utf8'),
                      comment_set, tvec, comment_tfidf)