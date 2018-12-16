"""
A PyTorch implementation of `A Structured Self-Attention Sentence Embedding`
https://arxiv.org/pdf/1703.03130.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAttention(nn.Module):
    def __init__(self, word_embedding, hidden_dim, num_classes):
        super(LSTMAttention, self).__init__()
        vocab_size = word_embedding.shape[0]
        embedding_dim = word_embedding.shape[1]
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim).from_pretrained(torch.Tensor(word_embedding), freeze=True)
        self.bilstm = nn.LSTM(input_size=embedding_dim,
                              hidden_size=hidden_dim // 2,
                              num_layers=1,
                              batch_first=False,
                              bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim, num_classes)
        self.self_attention = nn.Sequential(
            nn.Linear(hidden_dim, 100, bias=False),
            nn.LeakyReLU(),
            nn.Linear(100, 1, bias=False)
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, sentences):
        """LSTMAttention Model Forward.
        Arguments:
            sentences: Tensor with shape (batch_size, max_len)
        Returns:
            logits: Tensor with shape (batch_size, num_classes)

        """
        embeds = self.word_embedding(sentences)  # (N, L, C)

        embeds = embeds.permute(1, 0, 2)  # (L, N, C)
        lstm_out, _ = self.bilstm(embeds)  # (L, N, C)
        lstm_out = lstm_out.permute(1, 0, 2)  # (N, L, C)
        attention = self.self_attention(lstm_out)  # (N, L, 1)
        attention = attention.permute(0, 2, 1)  # (N, 1, L)
        attention = F.softmax(attention, dim=-1)  # (N, 1, L)
        feats = torch.bmm(attention, lstm_out).squeeze(dim=1)  # (N, C)
        feats = self.dropout(feats)
        logits = self.hidden2label(feats)  # (N, num_classes)

        return logits
