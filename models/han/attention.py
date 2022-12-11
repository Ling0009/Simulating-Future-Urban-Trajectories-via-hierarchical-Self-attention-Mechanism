import torch
from torch import nn


class HANAttention(nn.Module):
    """
    The word-level attention module.
    """

    def __init__(self, d_model, att_size):
        """
        :param d_model: size of (bidirectional) model
        :param att_size: size of word-level attention layer
        :param dropout: dropout
        """
        super(HANAttention, self).__init__()

        # Word-level attention network
        self.word_attention = nn.Linear(d_model, att_size)

        # Word context vector to take dot-product with
        self.word_context_vector = nn.Linear(att_size, 1, bias=False)
        # You could also do this with:
        # self.word_context_vector = nn.Parameter(torch.FloatTensor(1, word_att_size))
        # self.word_context_vector.data.uniform_(-0.1, 0.1)
        # And then take the dot-product


    def forward(self, sentences):
        """
        Forward propagation.

        :param sentences: encoded sentence-level data, a tensor of dimension (slice_num, slice_len, seq_len, d_model)
        :return: sentence embeddings, attention weights of words
        """

        # Find attention vectors by applying the attention linear layer on the output of the RNN
        att_w = self.word_attention(sentences)  # (slice_len, seq_len, att_size)
        att_w = torch.tanh(att_w)  # (slice_len, seq_len, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_w = self.word_context_vector(att_w).squeeze(2)  # (slice_len, seq_len)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence

        # First, take the exponent
        max_value = torch.max(att_w, dim=1)[0]  # scalar, for numerical stability during exponent calculation
        att_w = torch.exp(att_w - max_value)  # (slice_len, seq_len)

        # Calculate softmax values as now words are arranged in their respective sentences
        word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)  # (slice_len, seq_len)

        # Find sentence embeddings
        sentences = sentences * word_alphas.unsqueeze(2)  # (slice_len, seq_len, d_model)
        sentences = sentences.sum(dim=1)  # (slice_len, d_model)

        return sentences, word_alphas


class AddAttention(nn.Module):
    def __init__(self, d_traj, att_size):
        super(AddAttention, self).__init__()
        self.fc1 = nn.Linear(d_traj, att_size)
        self.fc2 = nn.Linear(att_size, d_traj)
        self.ac = nn.Tanh()

    def forward(self, history, present):
        history = self.fc1(history)
        present = self.fc1(present)
        return self.fc2(self.ac(history + present))

class PeriodicalAttention(nn.Module):
    def __init__(self, d_traj, att_size):
        super(PeriodicalAttention, self).__init__()
        self.attn = AddAttention(d_traj, att_size)
        self.fc = nn.Linear(5 * d_traj, d_traj)

    def forward(self, x):
        long = torch.concat([x[:24], self.attn(x[:-24], x[24:])])
        short_1 = torch.concat([x[:1], self.attn(x[:-1], x[1:])])
        short_2 = torch.concat([x[:2], self.attn(x[:-2], x[2:])])
        short_3 = torch.concat([x[:3], self.attn(x[:-3], x[3:])])
        short_4 = torch.concat([x[:4], self.attn(x[:-4], x[4:])])
        return self.fc(torch.concat([long, short_1, short_2, short_3, short_4], dim=-1))
