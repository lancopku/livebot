'''
 @Date  : 8/15/2018
 @Author: Shuming Ma
 @mail  : shumingma@pku.edu.cn 
 @homepage: shumingma.com
'''
import os
import json
import jieba
from PIL import Image
import pickle
from collections import OrderedDict
import random
from torchvision.transforms import functional as F
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.img_encoder = self.build_resnet()

    def build_resnet(self):
        resnet = torchvision.models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        resnet = nn.Sequential(*modules)
        for p in resnet.parameters():
            p.requires_grad = False
        return resnet

    def img_to_tensor(self, img):
        return F.normalize(F.to_tensor(img), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def encode(self, X):
        return self.img_encoder(X).squeeze(3).squeeze(2)

    def forward(self, X):
        X = self.img_to_tensor(X)
        X = Variable(X, volatile=True).unsqueeze(0).cuda()
        return self.encode(X).squeeze(0).data


def process(s):
    return list(jieba.cut(s.replace(' ','')))

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

class Dict(object):
    def __init__(self):
        self.word2id = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3, '<&&&>': 4}
        self.id2word = {0: '<PAD>', 1: '<BOS>', 2: '<EOS>', 3: '<UNK>', 4: '<&&&>'}
        self.frequency = {}

    def add(self, s):
        ids = []
        for w in s:
            if w in self.word2id:
                id = self.word2id[w]
                self.frequency[w] += 1
            else:
                id = len(self.word2id)
                self.word2id[w] = id
                self.id2word[id] = w
                self.frequency[w] = 1
            ids.append(id)
        return ids

    def transform(self, s):
        ids = []
        for w in s:
            if w in self.word2id:
                id = self.word2id[w]
            else:
                id = self.word2id['<UNK>']
            ids.append(id)
        return ids

    def prune(self, k):
        sorted_by_value = sorted(self.frequency.items(), key=lambda kv: -kv[1])
        newDict = Dict()
        newDict.add(list(zip(*sorted_by_value))[0][:k])
        return newDict

    def save(self, fout):
        return json.dump({'word2id': self.word2id, 'id2word': self.id2word}, fout, ensure_ascii=False)

    def load(self, fin):
        datas = json.load(fin)
        self.word2id = datas['word2id']
        self.id2word = datas['id2word']

    def __len__(self):
        return len(self.word2id)


def process_comments(filename, out_file, dicts=None):
    files = open(filename, 'r').read().strip().split('\n')
    datas = []
    print('starting %s' % filename)
    count = 0
    for file in files:
        lines = open(os.path.join('comment/', file), 'r', encoding='utf8').read().strip().split('\n')
        for line in lines:
            cols = line.split('\t')
            if len(cols) != 3:
                print(cols)
                continue
            comment = process(cols[2])
            if dicts is not None:
                dicts.add(comment)
            video = int(cols[0])
            time = int(cols[1])
            time = min(time, len(os.listdir('img/%d/' % video)))
            time = max(time, 1)
            datas.append({'video': video, 'time': time, 'comment': " ".join(comment)})
            count += 1
            if count % 100000 == 0:
                print('finishing %d' % count)

    if dicts is not None:
        dicts.save(open('dicts-whole.json', 'w', encoding='utf8'))
        dicts = dicts.prune(30000)
        dicts.save(open('dicts-30000.json', 'w', encoding='utf8'))

    dump_to_json(datas, open(out_file, 'w', encoding='utf8'))


def save_pretrain_imgs(indir):
    imgs = {}
    model = CNN()
    model.cuda()
    count = 0
    for i in range(len(os.listdir(indir))):
        imgs[i] = {}
        dir = os.path.join(indir, str(i))
        for j in range(len(os.listdir(dir))):
            img_path = os.path.join(dir, "%d.bmp" % (j + 1))
            img = Image.open(img_path)
            if img.size[0] == 224:
                imgs[i][j] = model(img)
                count += 1
            #print(imgs[i][j])
        print("%d/%d" % (i + 1, len(os.listdir(indir))))

    print(count)
    torch.save(imgs, open('res18.pkl', 'wb'))


def img_to_tensor(img):
    return F.normalize(F.to_tensor(img), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def train_comments(filename, outfile):
    datas = load_from_json(open(filename, 'r', encoding='utf8'))
    newdatas = []
    surroundings = [-1, 1, -2, 2, -3, 3, -4, 4]
    for i, data in enumerate(datas):
        context, limits = '', 5
        for t in surroundings:
            if i+t >= 0 and i+t < len(datas):
                if context == '':
                    context = datas[i+t]['comment']
                else:
                    context += ' <&&&> ' + datas[i+t]['comment']
        newdatas.append({'video': data['video'], 'time': data['time'],
                         'context': context, 'comment': data['comment']})

    dump_to_json(newdatas, open(outfile, 'w', encoding='utf8'))


def test_comments(filename, outfile, samples=5000):
    datas = load_from_json(open(filename, 'r', encoding='utf8'))
    dicts = {}
    for data in datas:
        key = data['video'] * 10000 + data['time']
        if key not in dicts:
            dicts[key] = [data['comment']]
        else:
            dicts[key].append(data['comment'])
    newdatas = []
    keys = list(dicts.keys())
    random.shuffle(keys)
    surroundings = [-1, 1, -2, 2, -3, 3]
    for key in keys:
        if len(newdatas) < samples and len(dicts[key]) >= 5:
            video, time, comments = key // 10000, key % 10000, dicts[key][:5]
            context, limits = '', 5
            for t in surroundings:
                if key+t in dicts:
                    for comment in dicts[key+t]:
                        if context == '':
                            context = comment
                            limits -= 1
                        else:
                            context += ' <&&&> ' + comment
                            limits -= 1
                            if limits == 0:
                                break
                    if limits == 0:
                        break
            newdatas.append({'video': video, 'time': time,
                             'context': context, 'comment': comments})
    print(outfile, len(newdatas))
    dump_to_json(newdatas, open(outfile, 'w', encoding='utf8'))


if __name__ == '__main__':
    save_pretrain_imgs('img/')
    process_comments('train.txt', 'train.json', Dict())
    process_comments('test.txt', 'test.json')
    process_comments('dev.txt', 'dev.json')
    train_comments('train.json', 'train-context.json')
    test_comments('test.json', 'test-context.json', samples=5000)
    test_comments('dev.json', 'dev-context.json', samples=8000)
