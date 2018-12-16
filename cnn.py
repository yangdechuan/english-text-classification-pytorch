"""
A PyTorch implementation of `Convolutional Neural Networks for Sentence Classification`
https://arxiv.org/pdf/1408.5882.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNNTextModel(nn.Module):
    def __init__(self, word_embedding, num_classes):
        super(CNNTextModel, self).__init__()
        vocab_size = word_embedding.shape[0]
        embedding_dim = word_embedding.shape[1]
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim).from_pretrained(torch.Tensor(word_embedding), freeze=False)
        self.conv13 = nn.Conv2d(1, 100, kernel_size=(3, embedding_dim))
        self.conv14 = nn.Conv2d(1, 100, kernel_size=(4, embedding_dim))
        self.conv15 = nn.Conv2d(1, 100, kernel_size=(5, embedding_dim))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, num_classes)

    def _conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(dim=3)  # (N, 100, L-i+1)
        x = F.max_pool1d(x, x.size(2)).squeeze(dim=2)  # (N, 100)
        return x

    def forward(self, sentences):
        """
        Forward function.
        Arguments:
            sentences: Tensor of shape (batch_size, max_len)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        embeds = self.word_embedding(sentences)  # (N, L, D)

        embeds = embeds.unsqueeze(1)  # (N, 1, L, D)

        cnn_out1 = self._conv_and_pool(embeds, self.conv13)  # (N, 100)
        cnn_out2 = self._conv_and_pool(embeds, self.conv14)  # (N, 100)
        cnn_out3 = self._conv_and_pool(embeds, self.conv15)  # (N, 100)

        cnn_out = torch.cat((cnn_out1, cnn_out2, cnn_out3), dim=1)  # (N, 300)
        cnn_out = self.dropout(cnn_out)
        logits = self.fc(cnn_out)  # (N, C)

        return logits
