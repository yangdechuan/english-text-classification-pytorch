import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import nltk
import gensim
from torch.utils.data import Dataset

PAD = 0


def _clean_str(string):
    string = re.sub("#SemST", "", string)
    # # # string = string.lower()
    # string = re.sub("[^A-Za-z0-9(),!?'`]", " ", string)
    # # string = re.sub("'s", " 's", string)
    # # string = re.sub("'ve", " 've", string)
    # # string = re.sub("n't", " n't", string)
    # # string = re.sub("'re", " 're", string)
    # # string = re.sub("'d", " 'd", string)
    # # string = re.sub("'ll", " 'll", string)
    # string = re.sub(",", " , ", string)
    # string = re.sub("!", " ! ", string)
    # string = re.sub("\(", " ( ", string)
    # string = re.sub("\)", " ) ", string)
    # string = re.sub("\?", " ? ", string)
    # string = re.sub("\s{2,}", " ", string)
    return string.strip()


def make_vocab(train_file, result_dir="results", text_col_name=None):
    """Build vocab dict.
    Write vocab and num to results/vocab.txt

    Arguments:
        train_file: train data file path.
        result_dir: vocab dict directory.
        text_col_name: column name for text.
    """
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    dic_filepath = os.path.join(result_dir, "vocab.txt")

    df = pd.read_csv(train_file)
    vocab2num = Counter()
    lengths = []
    for sentence in df[text_col_name]:
        sentence = _clean_str(sentence)
        # vocabs = sentence.split()
        vocabs = nltk.word_tokenize(sentence)
        lengths.append(len(vocabs))
        for vocab in vocabs:
            vocab = vocab.strip()
            if vocab != "":
                vocab2num[vocab] += 1
    with open(dic_filepath, "w", encoding="utf-8") as fw:
        fw.write("{}\t1000000000\n".format("<PAD>"))
        for vocab, num in vocab2num.most_common():
            fw.write("{}\t{}\n".format(vocab, num))

    print("Vocab Size {}".format(len(vocab2num)))
    print("Train Data Size {}".format(len(lengths)))
    print("Average Sentence Length {}".format(sum(lengths) / len(lengths)))
    print("Max Sentence Length {}".format(max(lengths)))


def load_data(file, max_len=100, min_count=1, result_dir="results", text_col_name=None, label_col_name=None):
    """Load texts and labels for train or test.
    Arguments:
        file: data file path.
        max_len: Sequences longer than this will be filtered out, and shorter than this will be padded with PAD.
        min_count: Vocab num less than this will be replaced with UNK.
        result_dir: vocab dict dir.
        text_col_name: column name for text.
        label_col_name: column name for label.
    Returns:
        X: int64 numpy array with shape (data_size, max_len)
        y: int64 numpy array with shape (data_size, ) or None
    """
    with open(os.path.join(result_dir, "vocab.txt"), "r", encoding="utf-8") as fr:
        vocabs = [line.split()[0] for line in fr.readlines() if int(line.split()[1]) >= min_count]
    vocab2idx = {vocab: idx for idx, vocab in enumerate(vocabs)}

    df = pd.read_csv(file)
    x_list = []
    for sentence in df[text_col_name]:
        sentence = _clean_str(sentence)
        x = [vocab2idx.get(vocab) for vocab in nltk.word_tokenize(sentence) if vocab in vocab2idx]
        x = x[: max_len]
        n_pad = max_len - len(x)
        x = x + n_pad * [PAD]
        x_list.append(x)
    X = np.array(x_list, dtype=np.int64)
    print("{} Data size {}".format("Train" if "train" in file else "Test", len(X)))

    y = df[label_col_name].values.astype(np.int64) if label_col_name in df.columns else None

    return X, y


def load_embedding(embedding_file, embedding_size=300, min_count=1, result_dir="results"):
    try:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=True)
    except Exception:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=False)
    with open(os.path.join(result_dir, "vocab.txt"), "r", encoding="utf-8") as fr:
        vocabs = [line.split()[0] for line in fr.readlines() if int(line.split()[1]) >= min_count]
    idx2vocab = {idx: vocab for idx, vocab in enumerate(vocabs)}
    vocab_size = len(idx2vocab)
    word_embedding = np.zeros((vocab_size, embedding_size), dtype=np.float32)
    for idx in range(1, vocab_size):
        try:
            vocab = idx2vocab[idx]
            word_embedding[idx] = word2vec[vocab]
        except KeyError:
            word_embedding[idx] = np.random.randn(embedding_size)
    return word_embedding


class CustomDataset(Dataset):
    """Custom Dataset for PyTorch."""
    def __init__(self, file,
                 max_len=100,
                 min_count=1,
                 result_dir="results",
                 text_col_name = "sentence",
                 label_col_name = "label"):
        super(CustomDataset, self).__init__()
        self.X, self.y = load_data(file,
                                   max_len=max_len,
                                   min_count=min_count,
                                   result_dir=result_dir,
                                   text_col_name=text_col_name,
                                   label_col_name=label_col_name)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]
