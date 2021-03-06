import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import nltk
import gensim

PAD = 0


def _clean_str(string, do_lower_case=True):
    if do_lower_case:
        string = string.lower()
    return string.strip()


def make_vocab(train_file, do_lower_case=True, result_dir="results", text_col_name=None):
    """Build vocab dict.
    Write vocab and num to results/vocab.txt

    Arguments:
        train_file: train data file path.
        do_lower_case: Whether to use lower case.
        result_dir: vocab dict directory.
        text_col_name: column name for text.
    """
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    dic_filepath = os.path.join(result_dir, "vocab.txt")

    df = pd.read_csv(train_file)
    vocab2num = Counter()
    lengths = []
    print("Making vocab...")
    for sentence in df[text_col_name]:
        sentence = _clean_str(sentence, do_lower_case=do_lower_case)
        # vocabs = sentence.split()
        # vocabs = nltk.word_tokenize(sentence)
        vocabs = nltk.wordpunct_tokenize(sentence)
        lengths.append(len(vocabs))
        for vocab in vocabs:
            vocab = vocab.strip()
            vocab2num[vocab] += 1
    with open(dic_filepath, "w", encoding="utf-8") as fw:
        fw.write("{}\t1000000000\n".format("<PAD>"))
        for vocab, num in vocab2num.most_common():
            fw.write("{}\t{}\n".format(vocab, num))
    print("Finish making vocab!")

    print("Vocab Size of all train data {}".format(len(vocab2num)))
    print("Train Data Size {}".format(len(lengths)))
    print("Average Sentence Length {}".format(sum(lengths) / len(lengths)))
    print("Max Sentence Length {}".format(max(lengths)))


def get_vocab(result_dir="results", min_count=1):
    with open(os.path.join(result_dir, "vocab.txt"), "r", encoding="utf-8") as fr:
        vocabs = [line.split()[0] for line in fr.readlines() if int(line.split()[1]) >= min_count]
    vocab2idx = {vocab: idx for idx, vocab in enumerate(vocabs)}
    print("used vocab size {}".format(len(vocab2idx)))
    return vocab2idx


def load_data(file,
              max_len=100,
              vocab2idx=None,
              do_lower_case=True,
              text_col_name=None,
              label_col_name=None,
              class_names=None):
    """Load texts and labels for train or test.
    Arguments:
        file: data file path.
        max_len: Sequences longer than this will be filtered out, and shorter than this will be padded with PAD.
        vocab2idx: dict. e.g. {"hello": 1, "world": 7, ...}
        do_lower_case: Whether to use lower case.
        text_col_name: column name for text.
        label_col_name: column name for label.
        class_names: list of label name.
    Returns:
        (X, y)
        X: int64 numpy array with shape (data_size, max_len)
        y: int64 numpy array with shape (data_size, ) or None
            If label_col_name is not None, y is numpy array.
            If label_col_name is None, y is None.
    """
    df = pd.read_csv(file)
    x_list = []
    y_list = []
    if label_col_name:
        label2idx = {label: idx for idx, label in enumerate(class_names)}
    for i in range(df.shape[0]):
        if label_col_name and df[label_col_name][i] not in class_names:
            continue
        if label_col_name:
            label = df[label_col_name][i]
            y_list.append(label2idx[label])
        sentence = df[text_col_name][i]
        sentence = _clean_str(sentence, do_lower_case=do_lower_case)
        x = [vocab2idx.get(vocab) for vocab in nltk.wordpunct_tokenize(sentence) if vocab in vocab2idx]
        x = x[: max_len]
        n_pad = max_len - len(x)
        x = x + n_pad * [PAD]
        x_list.append(x)

    X = np.array(x_list, dtype=np.int64)
    y = np.array(y_list, dtype=np.int64) if y_list else None
    print("{} Data size {}".format("Train" if "train" in file else "Test", len(X)))

    return X, y


def load_embedding(embedding_file, vocab2idx):
    if embedding_file[-4: ] == ".bin":
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=True)
    else:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=False)
    vocab_size = len(vocab2idx)
    embedding_size = word2vec.vector_size
    word_embedding = np.zeros((vocab_size, embedding_size), dtype=np.float32)
    idx2vocab = {idx: vocab for vocab, idx in vocab2idx.items()}

    for idx in range(1, vocab_size):
        vocab = idx2vocab[idx]
        try:
            word_embedding[idx] = word2vec[vocab]
        except KeyError:
            word_embedding[idx] = np.random.randn(embedding_size)

    return word_embedding

