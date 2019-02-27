import codecs

def read_train(file_name):
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
    return sentences

def read_test(file_name):
    f = codecs.open(file_name, encoding='utf-8')
    lines = f.readlines()
    sentences = []
    for i,line in enumerate(lines):
        sentence = []
        e1 = -1
        e2 = -1
        sp = line[line.find('"')+1:-4].split()
        for j,w in enumerate(sp):
            print(w)
            if w.startswith('<e1>'):
                sentence.append(w[4:-5])
                e1 = j
            elif w.startswith('<e2>'):
                sentence.append(w[4:-5])
                e2 = j
            else:
                sentence.append(w)
        sentences.append([sentence, e1, e2])
    return sentences

if __name__ == '__main__':
    sentences = read_train('data/train_file.txt')
    #print(sentences)
    test_sentences = read_test('data/test_file.txt')
    print(test_sentences)

