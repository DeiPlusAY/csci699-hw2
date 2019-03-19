import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        # dimension of hidden nodes
        self.dim_pos = args.dim_pos
        self.dim_word = args.dim_word
        if args.bert:
            self.dim_word = 768
        self.dim_conv = args.dim_conv
        self.bert = args.bert

        # shape of dicts
        #self.len_pos = args.len_pos
        if not args.bert:
            self.len_word = args.len_word
        self.len_rel = args.len_rel
        
        # sequence
        self.len_seq = args.len_seq

        # self layers
        if not args.bert:
            self.word_embedding = nn.Embedding(self.len_word, self.dim_word)
            self.word_embedding.weight.data.copy_(torch.from_numpy(args.word_embedding))
            self.word_embedding.requires_grad = False
        self.pos_embedding = nn.Embedding(3, self.dim_pos)
        self.dropout = nn.Dropout(args.dropout_rate)

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(self.dim_word + self.dim_pos, self.dim_conv, kernel_size=kernel_size, padding=int((kernel_size) / 2)),
            nn.ReLU(),
            nn.MaxPool1d(self.len_seq)
        ) for kernel_size in args.kernel_sizes])

        self.pool = nn.MaxPool1d(self.len_seq)
        '''
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, self.dim_conv, (kernel_size, self.dim_word+self.dim_pos), padding=(int(kernel_size / 2) , 0)),
            nn.ReLU()
        )for kernel_size in args.kernel_sizes])
        '''
        self.fc = nn.Linear(self.dim_conv * len(self.convs) + 2 * self.dim_word, self.len_rel)
    
    def forward(self, W, W_pos, e1, e2):
        if self.bert:
            e1_emb = e1
            e2_emb = e2
        else:
            e1_emb = self.word_embedding(e1)
            e2_emb = self.word_embedding(e2)

        if not self.bert:
            W = self.word_embedding(W)
        W_pos = self.pos_embedding(W_pos)
        Wa = torch.cat([W, W_pos], dim=2)
        conv = [conv_f(Wa.permute((0,2,1))) for conv_f in self.convs]
        #conv = [conv_f(Wa.unsqueeze(1)).squeeze(3) for conv_f in self.convs]
        #conv = [self.pool(i) for i in conv]
        #print([c.shape for c in conv])
        conv = torch.cat(conv, dim=1)

        conv = self.dropout(conv)
        e_concat = torch.cat([e1_emb, e2_emb], dim=1).float()
        all_concat = torch.cat([e_concat.view(e_concat.size(0), -1), conv.view(conv.size(0), -1)], dim=1)
        out = self.fc(all_concat)
        out = F.softmax(out, dim=1)

        return out

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)

class BERTBaseModel(nn.Module):
    def __init__(self, args):
        super(BERTBaseModel, self).__init__()
        if args.dim_bert > 0:
            self.dense1 = nn.Linear(768,args.dim_bert)
            self.dense2 = nn.Linear(args.dim_bert,args.len_rel)
        else:
            self.dense1 = nn.Linear(768,args.len_rel)
        self.dim_bert = args.dim_bert
    def forward(self, w):
        x = self.dense1(w)
        if self.dim_bert > 0:
            x = F.relu(x)
            x = self.dense2(x)
        out = F.softmax(x, dim=1)
        return out
