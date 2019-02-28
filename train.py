import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import CNN
from read import *

def accuracy(preds, labels):
    total = preds.size(0)
    preds = torch.max(preds, dim=1)[1]
    correct = (preds == labels).sum()
    acc = correct.cpu().item() / total
    return acc

def main():
    parser = argparse.ArgumentParser("CNN")
    parser.add_argument("--dp", dest="dim_pos", type=int, default=5)
    parser.add_argument("--dc", dest="dim_conv", type=int, default=32)
    parser.add_argument("--dw", dest="dim_word", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l2", type=float, default=0.0005)
    parser.add_argument("--l1", type=float, default=0.05)
    parser.add_argument("--len_seq", type=int, default=85)
    parser.add_argument("--len_rel", type=int, default=19)
    parser.add_argument("--epoch", type=int ,default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--dropout_rate", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--kernel_sizes", type=str, default="3,4,5")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--train_filename', default='./data/train_file.txt')
    parser.add_argument('--val_filename', default='./data/test_full.txt')
    parser.add_argument('--test_filename', default='./data/test_file.txt')
    parser.add_argument('--model_file', default='./cnn.pt')
    parser.add_argument('--embedding', default='/data/hejiang/glove/glove.6B.100d.txt')
    parser.add_argument('--bert', type=bool, default=False)
    parser.add_argument('--optimizer',  default='adam')
    

    args = parser.parse_args()
    args.kernel_sizes = list(map(int, args.kernel_sizes.split(',')))
    print(args)
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    args.word_to_idx = None
    args.word_embedding, args.word_to_idx = loadGloveModel(args.embedding)

    dataset = SemEvalDataset(args.train_filename, word_to_idx=args.word_to_idx, max_len=args.len_seq)
    dataloader = DataLoader(dataset, args.batch_size, True, num_workers=args.num_workers)
    dataset_val = SemEvalDataset(args.val_filename, word_to_idx=dataset.word_to_idx, tag_to_idx=dataset.tag_to_idx, max_len=args.len_seq)
    dataloader_val = DataLoader(dataset_val, args.batch_size, True, num_workers=args.num_workers)

    args.len_word = len(dataset.word_to_idx)
    args.len_rel = len(dataset.tag_to_idx)

    if args.gpu >= 0:
        model = CNN(args).cuda()
    else:
        model = CNN(args)
    model = train(args, dataloader, dataloader_val, model)
    eval(args, dataloader_val, model)

def train(args, dataloader, dataloader_val, model):
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),lr=args.lr, weight_decay=args.l2)
    loss_func = nn.CrossEntropyLoss()

    best_eval_acc = 0.
    # Training
    total_loss = 0.
    total_acc = 0.
    ntrain_batch = 0
    model.train()
    for i in range(args.epoch):
        for (seq, w_pos, e1, e2, r) in dataloader:
            ntrain_batch += 1
            if args.gpu >= 0:
                seq = Variable(seq).cuda()
                e1 = Variable(e1).cuda()
                e2 = Variable(e2).cuda()
                w_pos = Variable(w_pos).cuda()
                r = Variable(r).cuda()
            else:
                seq = Variable(seq)
                e1 = Variable(e1)
                e2 = Variable(e2)
                w_pos = Variable(w_pos)
                r = Variable(r)
            
            r = r.view(r.size(0))

            pred = model(seq, w_pos, e1, e2)
            l = loss_func(pred, r)
            acc = accuracy(pred, r)
            total_acc += acc
            total_loss += l.item()

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        print("Epoch: {}, Training loss : {:.4}, acc: {:.4}".\
        format(i, total_loss/ntrain_batch, total_acc / ntrain_batch))
        if (i+1) % args.eval_every == 0:
            eval(args, dataloader_val, model)
    return model

def eval(args, dataloader, model):
    loss_func = nn.CrossEntropyLoss()
    
    # Training
    total_loss = 0.
    total_acc = 0.
    ntrain_batch = 0
    model.eval()
    for (seq, w_pos, e1, e2, r) in dataloader:
        ntrain_batch += 1
        if args.gpu >= 0:
            seq = Variable(seq).cuda()
            e1 = Variable(e1).cuda()
            e2 = Variable(e2).cuda()
            w_pos = Variable(w_pos).cuda()
            r = Variable(r).cuda()
        else:
            seq = Variable(seq)
            e1 = Variable(e1)
            e2 = Variable(e2)
            w_pos = Variable(w_pos)
            r = Variable(r)
        
        r = r.view(r.size(0))

        pred = model(seq, w_pos, e1, e2)
        l = loss_func(pred, r)
        acc = accuracy(pred, r)
        total_acc += acc
        total_loss += l.item()

    print("Val loss : {:.4}, acc: {:.4}".\
    format(total_loss/ntrain_batch, total_acc / ntrain_batch))


if __name__ == '__main__':
    main()
