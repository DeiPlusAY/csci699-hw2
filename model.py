import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        # dimension of hidden nodes
        self.dim_pos = args.dim_pos
        self.dim_word = args.dim_word
        self.dim_conv = args.dim_conv
        self.bert = args.bert

        # shape of dicts
        #self.len_pos = args.len_pos
        self.len_word = args.len_word
        self.len_rel = args.len_rel
        
        # sequence
        self.len_seq = args.len_seq

        # self layers
        self.word_embedding = nn.Embedding(self.len_word, self.dim_word)
        self.word_embedding.weight.data.copy_(torch.from_numpy(args.word_embedding))
        self.pos_embedding = nn.Embedding(3, self.dim_pos)
        self.dropout = nn.Dropout(args.dropout_rate)

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(self.dim_word + self.dim_pos, self.dim_conv, kernel_size=kernel_size, padding=int((kernel_size) / 2)),
            nn.Tanh(),
            nn.MaxPool1d(self.len_seq)
        ) for kernel_size in args.kernel_sizes])

        self.fc = nn.Linear(self.dim_conv * len(self.convs) + 2 * self.dim_word, self.len_rel)
    
    def forward(self, W, W_pos, e1, e2):
        if self.bert:
            e1_emb = W[:,e1]
            e2_emb = W[:,e2]
        else:
            e1_emb = self.word_embedding(e1)
            e2_emb = self.word_embedding(e2)

        W = self.word_embedding(W)        
        W_pos = self.pos_embedding(W_pos)
        Wa = torch.cat([W, W_pos], dim=2)

        conv = [conv(Wa.permute(0, 2, 1)) for conv in self.convs]
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
