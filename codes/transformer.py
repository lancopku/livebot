'''
 @Date  : 8/22/2018
 @Author: Shuming Ma
 @mail  : shumingma@pku.edu.cn
 @homepage: shumingma.com
'''

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as Optim
from torch.autograd import Variable
import torchvision
from torchvision.transforms import functional as F

import json
import argparse
import time
import os
import random
import pickle
from PIL import Image
import numpy as np
from metrics import *
from modules import *

parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('-n_emb', type=int, default=512, help="Embedding size")
parser.add_argument('-n_hidden', type=int, default=512, help="Hidden size")
parser.add_argument('-d_ff', type=int, default=2048, help="Hidden size of Feedforward")
parser.add_argument('-n_head', type=int, default=8, help="Number of head")
parser.add_argument('-n_block', type=int, default=6, help="Number of block")
parser.add_argument('-batch_size', type=int, default=64, help="Batch size")
parser.add_argument('-vocab_size', type=int, default=30000, help="Vocabulary size")
parser.add_argument('-epoch', type=int, default=50, help="Number of epoch")
parser.add_argument('-report', type=int, default=500, help="Number of report interval")
parser.add_argument('-lr', type=float, default=3e-4, help="Learning rate")
parser.add_argument('-dropout', type=float, default=0.1, help="Dropout rate")
parser.add_argument('-restore', type=str, default='', help="Restoring model path")
parser.add_argument('-mode', type=str, default='train', help="Train or test")
parser.add_argument('-dir', type=str, default='ckpt', help="Checkpoint directory")
parser.add_argument('-max_len', type=int, default=20, help="Limited length for text")
parser.add_argument('-n_img', type=int, default=5, help="Number of input images")
parser.add_argument('-n_com', type=int, default=5, help="Number of input comments")

opt = parser.parse_args()

data_path = 'data/'
train_path, test_path, dev_path = data_path + 'train-context.json', data_path + 'test-candidate.json', data_path + 'dev-candidate.json'
vocab_path = data_path + 'dicts-30000.json'
img_path = data_path + 'res18.pkl'

vocabs = json.load(open(vocab_path, 'r', encoding='utf8'))['word2id']
rev_vocabs = json.load(open(vocab_path, 'r', encoding='utf8'))['id2word']
opt.vocab_size = len(vocabs)

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

if not os.path.exists(opt.dir):
    os.mkdir(opt.dir)

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


class Model(nn.Module):

    def __init__(self, n_emb, n_hidden, vocab_size, dropout, d_ff, n_head, n_block):
        super(Model, self).__init__()
        self.n_emb = n_emb
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.embedding = nn.Sequential(Embeddings(n_hidden, vocab_size), PositionalEncoding(n_hidden, dropout))
        self.video_encoder = VideoEncoder(n_hidden, d_ff, n_head, dropout, n_block)
        self.text_encoder = TextEncoder(n_hidden, d_ff, n_head, dropout, n_block)
        self.comment_decoder = CommentDecoder(n_hidden, d_ff, n_head, dropout, n_block)
        self.output_layer = nn.Linear(self.n_hidden, self.vocab_size)
        self.criterion = nn.CrossEntropyLoss(reduce=False, size_average=False, ignore_index=0)

    def encode_img(self, X):
        out = self.video_encoder(X)
        return out

    def encode_text(self, X, m):
        embs = self.embedding(X)
        out = self.text_encoder(embs, m)
        return out

    def decode(self, x, m1, m2, mask):
        embs = self.embedding(x)
        out = self.comment_decoder(embs, m1, m2, mask)
        out = self.output_layer(out)
        return out

    def forward(self, X, Y, T):
        out_img = self.encode_img(X)
        out_text = self.encode_text(T, out_img)
        mask = Variable(subsequent_mask(Y.size(0), Y.size(1)-1), requires_grad=False).cuda()
        outs = self.decode(Y[:,:-1], out_img, out_text, mask)

        Y = Y.t()
        outs = outs.transpose(0, 1)

        loss = self.criterion(outs.contiguous().view(-1, self.vocab_size),
                              Y[1:].contiguous().view(-1))

        return torch.mean(loss)


    def generate(self, X, T):
        out_img = self.encode_img(X)
        out_text = self.encode_text(T, out_img)

        ys = torch.ones(X.size(0), 1).long()
        for i in range(opt.max_len):
            out = self.decode(Variable(ys, volatile=True).cuda(), out_img, out_text,
                              Variable(subsequent_mask(ys.size(0), ys.size(1))))
            prob = out[:, -1]
            _, next_word = torch.max(prob, dim=-1, keepdim=True)
            next_word = next_word.data
            ys = torch.cat([ys, next_word], dim=-1)

        return ys[:, 1:]


    def ranking(self, X, Y, T):
        nums = len(Y)
        out_img = self.encode_img(X.unsqueeze(0))
        out_text = self.encode_text(T.unsqueeze(0), out_img)
        out_img = out_img.repeat(nums, 1, 1)
        out_text = out_text.repeat(nums, 1, 1)

        mask = Variable(subsequent_mask(Y.size(0), Y.size(1) - 1), requires_grad=False).cuda()
        outs = self.decode(Y[:, :-1], out_img, out_text, mask)

        Y = Y.t()
        outs = outs.transpose(0, 1)

        loss = self.criterion(outs.contiguous().view(-1, self.vocab_size),
                              Y[1:].contiguous().view(-1))

        loss = loss.view(-1, nums).sum(0)
        return torch.sort(loss, dim=0, descending=True)[1]



class DataSet(torch.utils.data.Dataset):

    def __init__(self, data_path, vocabs, is_train=True, imgs=None):
        print("starting load...")
        start_time = time.time()
        self.datas = load_from_json(open(data_path, 'r', encoding='utf8'))
        if imgs is not None:
            self.imgs = imgs
        else:
            self.imgs = torch.load(open(img_path, 'rb'))
        print("loading time:", time.time() - start_time)

        self.vocabs = vocabs
        self.vocab_size = len(self.vocabs)
        self.is_train = is_train

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        video_id, video_time = data['video'], data['time']-1

        X = self.load_imgs(video_id, video_time)
        T = self.load_comments(data['context'])

        if not self.is_train:
            comment = data['comment'][0]
        else:
            comment = data['comment']
        Y = DataSet.padding(comment, opt.max_len)

        return X, Y, T

    def get_img_and_candidate(self, index):
        data = self.datas[index]
        video_id, video_time = data['video'], data['time']

        X = self.load_imgs(video_id, video_time)
        T = self.load_comments(data['context'])

        Y = [DataSet.padding(c, opt.max_len) for c in data['candidate']]
        return X, torch.stack(Y), T, data

    def load_imgs(self, video_id, video_time):
        if opt.n_img == 0:
            return torch.stack([self.imgs[video_id][video_time].fill_(0.0) for _ in range(5)])

        surroundings = [0, -1, 1, -2, 2, -3, 3, -4, 4]
        X = []
        for t in surroundings:
            if video_time + t >= 0 and video_time + t < len(self.imgs[video_id]):
                X.append(self.imgs[video_id][video_time + t])
                if len(X) == opt.n_img:
                    break
        return torch.stack(X)

    def load_comments(self, context):
        if opt.n_com == 0:
            return torch.LongTensor([1]+[0]*opt.max_len*5+[2])
        return DataSet.padding(context, opt.max_len*opt.n_com)

    @staticmethod
    def padding(data, max_len):
        data = data.split()
        if len(data) > max_len-2:
            data = data[:max_len-2]
        Y = list(map(lambda t: vocabs.get(t, 3), data))
        Y = [1] + Y + [2]
        length = len(Y)
        Y = torch.cat([torch.LongTensor(Y), torch.zeros(max_len - length).long()])
        return Y

    @staticmethod
    def transform_to_words(ids):
        words = []
        for id in ids:
            if id == 2:
                break
            words.append(rev_vocabs[str(id.item())])
        return "".join(words)


def get_dataset(data_path, vocabs, is_train=True, imgs=None):
    return DataSet(data_path, vocabs, is_train=is_train, imgs=imgs)

def get_dataloader(dataset, batch_size, is_train=True):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_train)

def save_model(path, model):
    model_state_dict = model.state_dict()
    torch.save(model_state_dict, path)


def train():
    train_set = get_dataset(train_path, vocabs, is_train=True)
    dev_set = get_dataset(dev_path, vocabs, is_train=False, imgs=train_set.imgs)
    train_batch = get_dataloader(train_set, opt.batch_size, is_train=True)
    model = Model(n_emb=opt.n_emb, n_hidden=opt.n_hidden, vocab_size=opt.vocab_size,
                  dropout=opt.dropout, d_ff=opt.d_ff, n_head=opt.n_head, n_block=opt.n_block)
    if opt.restore != '':
        model_dict = torch.load(opt.restore)
        model.load_state_dict(model_dict)
    model.cuda()
    optim = Optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=opt.lr)
    best_score = -1000000

    for i in range(opt.epoch):
        model.train()
        report_loss, start_time, n_samples = 0, time.time(), 0
        count, total = 0, len(train_set) // opt.batch_size + 1
        for batch in train_batch:
            model.zero_grad()
            X, Y, T = batch
            X = Variable(X).cuda()
            Y = Variable(Y).cuda()
            T = Variable(T).cuda()
            loss = model(X, Y, T)
            loss.backward()
            optim.step()
            report_loss += loss.data[0]
            n_samples += len(X.data)
            count += 1
            if count % opt.report == 0 or count == total:
                print('%d/%d, epoch: %d, report_loss: %.3f, time: %.2f'
                      % (count, total, i+1, report_loss / n_samples, time.time() - start_time))
                model.eval()
                score = eval(dev_set, model)
                model.train()
                if score > best_score:
                    best_score = score
                    save_model(os.path.join(opt.dir, 'best_checkpoint.pt'), model)
                else:
                    save_model(os.path.join(opt.dir, 'checkpoint.pt'), model)
                report_loss, start_time, n_samples = 0, time.time(), 0

    return model


def eval(dev_set, model):
    print("starting evaluating...")
    start_time = time.time()
    model.eval()
    predictions, references = [], []
    dev_batch = get_dataloader(dev_set, opt.batch_size, is_train=False)

    loss = 0
    for batch in dev_batch:
        X, Y, T = batch
        X = Variable(X, volatile=True).cuda()
        Y = Variable(Y, volatile=True).cuda()
        T = Variable(T, volatile=True).cuda()
        loss += model(X, Y, T).data[0]

    print(loss)
    print("evaluting time:", time.time() - start_time)

    return -loss


def test(test_set, model):
    print("starting testing...")
    start_time = time.time()
    model.eval()
    predictions, references = [], []

    for i in range(len(test_set)):
        X, Y, T, data = test_set.get_img_and_candidate(i)
        X = Variable(X, volatile=True).cuda()
        Y = Variable(Y, volatile=True).cuda()
        T = Variable(T, volatile=True).cuda()
        ids = model.ranking(X, Y, T).data

        candidate = []
        comments = list(data['candidate'].keys())
        for id in ids:
            candidate.append(comments[id])
        predictions.append(candidate)
        references.append(data['candidate'])
        if i % 100 == 0:
            print(i)

    recall_1 = recall(predictions, references, 1)
    recall_5 = recall(predictions, references, 5)
    recall_10 = recall(predictions, references, 10)
    mr = mean_rank(predictions, references)
    mrr = mean_reciprocal_rank(predictions, references)
    print(recall_1, recall_5, recall_10, mr, mrr)

    print("testing time:", time.time() - start_time)



if __name__ == '__main__':
    if opt.mode == 'train':
        train()
    else:
        test_set = get_dataset(test_path, vocabs, is_train=False)
        model = Model(n_emb=opt.n_emb, n_hidden=opt.n_hidden, vocab_size=opt.vocab_size,
                  dropout=opt.dropout, d_ff=opt.d_ff, n_head=opt.n_head, n_block=opt.n_block)
        model_dict = torch.load(opt.restore)
        model.load_state_dict(model_dict)
        model.cuda()
        test(test_set, model)
