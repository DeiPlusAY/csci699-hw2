import codecs
import numpy as np
import torch
import torch.utils.data

def read_train(file_name):
    print("Loading Train Sentences")
    f = codecs.open(file_name, encoding='utf-8')
    lines = f.readlines()
    sentences = []
    sentence = []
    e1 = -1
    e2 = -1
    for i,line in enumerate(lines):
        if i % 4 == 0:
            sp = line[6:-4].split()
            e1,e2 = -1,-1
            sentence = []
            for j,w in enumerate(sp):
                # Recognize entity marker and seperate
                if w.startswith('<e1>'):
                    sentence.append(w[4:-5])
                    e1 = j
                elif w.startswith('<e2>'):
                    sentence.append(w[4:-5])
                    e2 = j
                else:
                    sentence.append(w)
        elif i % 4 == 1:
            tag = line[:-2]
            sentences.append([sentence,e1,e2,tag])
        else:
            pass
    print("Done.", len(sentences), " train sentences loaded!")
    return sentences

def read_test(file_name):
    print("Loading Test Sentences")
    f = codecs.open(file_name, encoding='utf-8')
    lines = f.readlines()
    sentences = []
    for i,line in enumerate(lines):
        sentence = []
        e1 = -1
        e2 = -1
        sp = line[line.find('"')+1:-4].split()
        for j,w in enumerate(sp):
            if w.startswith('<e1>'):
                sentence.append(w[4:-5])
                e1 = j
            elif w.startswith('<e2>'):
                sentence.append(w[4:-5])
                e2 = j
            else:
                sentence.append(w)
        sentences.append([sentence, e1, e2])
    print("Done.",len(sentences), " test sentences loaded!")
    return sentences

def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = codecs.open(gloveFile,'r',encoding='utf-8').readlines()
    model = np.zeros((len(f) + 1, 100))
    word_to_idx = {}
    word_to_idx['<UNK>'] = 0
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        word_to_idx[word] = len(word_to_idx)
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word_to_idx[word]] = embedding
    print ("Done.",len(model)," words loaded!")

    return model, word_to_idx

class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, tag_to_idx=None, train=True):
        self.train_set = read_train(file_name)
        from bert_serving.client import BertClient
        bc = BertClient()
        if tag_to_idx:
            self.tag_to_idx = tag_to_idx
        else:
            self.build_tag_dict()
        sens = [' '.join(s[0]) for s in self.train_set]
        sens_vec = bc.encode(sens)
        tags = np.zeros((len(sentences), 1))
        for i,s in enumerate(self.train_set):
            self.train_set[i][0] = sens_vec[i,:]
            tags[i] = self.tag_to_idx[s[-1]]
        self.tags = tags

    def build_tag_dict(self):
        self.tag_to_idx = {}
        for sentence in self.train_set:
            tag = sentence[-1]
            if tag not in self.tag_to_idx:
                self.tag_to_idx[tag] = len(self.tag_to_idx)
    
    def __getitem__(self, index):
        w = torch.from_numpy(self.train_set[index][0])
        r = torch.from_numpy(self.tags[index]).long()
        return w, r

    def __len__(self):
        return len(self.train_set)

class SemEvalDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, max_len = 85, word_to_idx=None, tag_to_idx=None, train=True):
        if train:
            self.train_set = read_train(file_name)
        else:
            self.train_set = read_test(file_name)

        self.max_len = max_len
        if word_to_idx != None:
            self.word_to_idx = word_to_idx
        else:
            self.build_dict()
        if tag_to_idx != None:
            self.tag_to_idx = tag_to_idx
        else:
            self.build_tag_dict()

        if train == True:
            self.seqs, self.w_poss, self.e1s, self.e2s, self.tags = self.vectorize_seq(self.train_set)
        else:
            self.seqs, self.w_poss, self.e1s, self.e2s = self.vectorize_test(self.train_set)

    def build_dict(self):
        print("Build Word Dict")
        self.word_to_idx = {}
        self.word_to_idx['<PAD>'] = 0
        self.word_to_idx['<UNK>'] = 0
        for sentence in self.train_set:
            for w in sentence[0]:
                if w not in self.word_to_idx:
                    self.word_to_idx[w] = len(self.word_to_idx)
        print("Done.", len(self.word_to_idx), "building word dict.")

    def build_tag_dict(self):
        self.tag_to_idx = {}
        for sentence in self.train_set:
            tag = sentence[-1]
            if tag not in self.tag_to_idx:
                self.tag_to_idx[tag] = len(self.tag_to_idx)

    def vectorize_seq(self, sentences):
        seqs = np.zeros((len(sentences), self.max_len))
        w_poss = np.zeros((len(sentences), self.max_len))
        e1s = np.zeros((len(sentences), 1))
        e2s = np.zeros((len(sentences), 1))
        tags = np.zeros((len(sentences), 1))
        for r, (words, e1, e2, tag) in enumerate(sentences):
            vec_words = list(map(lambda x: self.word_to_idx[x] if x in self.word_to_idx else 0, words))
            w_poss[r, e1] = 1
            w_poss[r, e2] = 2
            e1s[r] = vec_words[e1]
            e2s[r] = vec_words[e2]
            for i in range(min(self.max_len, len(words))):
                seqs[r, i] = vec_words[i]
            tags[r] = self.tag_to_idx[tag]
        return seqs, w_poss, e1s, e2s, tags
    
    def vectorize_test(self, sentences):
        seqs = np.zeros((len(sentences), self.max_len))
        w_poss = np.zeros((len(sentences), self.max_len))
        e1s = np.zeros((len(sentences), 1))
        e2s = np.zeros((len(sentences), 1))
        for r, (words, e1, e2) in enumerate(sentences):
            vec_words = list(map(lambda x: self.word_to_idx[x] if x in self.word_to_idx else 0, words))
            w_poss[r, e1] = 1
            w_poss[r, e2] = 2
            e1s[r] = vec_words[e1]
            e2s[r] = vec_words[e2]
            for i in range(min(self.max_len, len(words))):
                seqs[r, i] = vec_words[i]
        return seqs, w_poss, e1s, e2s

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = torch.from_numpy(self.seqs[index]).long()
        w_pos = torch.from_numpy(self.w_poss[index]).long()
        e1 = torch.from_numpy(self.e1s[index]).long()
        e2 = torch.from_numpy(self.e2s[index]).long()
        r = torch.from_numpy(self.tags[index]).long()
        return seq, w_pos, e1, e2, r
    
    def get_max_len(self):
        max_len = 0
        for sentence in self.train_set:
            if len(sentence[0]) > max_len:
                max_len = len(sentence[0])
        return max_len

def get_tags(sentences):
    return [s[-1] for s in sentences]

def write_preds(preds, output_path):
    o = codecs.open(output_path,'w',encoding='utf-8')
    for pred in preds:
        o.write(pred + '\n')

if __name__ == '__main__':
    #sentences = read_train('data/train_file.txt')
    #print(sentences)
    #test_sentences = read_test('data/test_file.txt')
    #print(test_sentences)
    #model = loadGloveModel('/data/glove/glove.6B.100d.txt')
    #dataset = SemEvalDataset('data/train_file.txt', max_len=85)
    #print(dataset.get_max_len())
    bert_dataset = BERTDataset('data/train_file.txt')
    print(bert_dataset.train_set)