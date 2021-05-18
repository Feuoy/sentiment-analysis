import torch.nn as nn
import torch.nn.functional as F
import torch
from base.base_model import BaseModel
from utils.model_utils import prepare_pack_padded_sequence
from transformers import BertModel


class MLP(BaseModel):

    def __init__(self, hidden_dim, output_dim, dropout, word_embedding, freeze,
                 needed_by_mlp_num_hidden_layers, needed_by_mlp_max_seq_len):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding.vectors), freeze=freeze)
        self.embedding.weight.data[1] = torch.zeros(self.embedding_dim)
        self.embedding_dim = word_embedding.vectors.shape[1]
        self.num_hidden_layers = needed_by_mlp_num_hidden_layers
        self.max_seq_len = needed_by_mlp_max_seq_len

        self.one_hidden_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.all_hidden_layer = nn.Sequential(self.one_hidden_layer)
        for i in range(self.num_hidden_layers - 2):
            self.all_hidden_layer.add_module(str(i + 1), self.one_hidden_layer)
        if self.num_hidden_layers > 1:
            self.mlp = nn.Sequential(
                nn.Linear(self.max_seq_len * self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                self.all_hidden_layer,
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.max_seq_len * self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.output_dim),
            )

    def forward(self, text, _, text_lengths):
        # text [batch_size, seq_len]
        embedded = self.embedding(text).float()
        # embedded [batch_size, seq_len, emb_dim]
        embedded_ = embedded.view(embedded.size(0), -1)
        # embedded_ [batch_size, seq_len * emb_dim]

        out = self.mlp(embedded_)
        # out [batch_size, output_dim]

        return out


class TextCNN1d(BaseModel):

    def __init__(self, n_filters, filter_sizes, output_dim, dropout, word_embedding, freeze):
        super().__init__()

        self.embedding_dim = word_embedding.vectors.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding.vectors), freeze=freeze)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, _, text_lengths):
        # text [batch_size, seq_len]
        embedded = self.embedding(text).float()
        # embedded [batch_size, seq_len, emb_dim]
        embedded = embedded.permute(0, 2, 1)
        # embedded [batch_size, emb_dim, seq_len]

        conved = [
            F.relu(conv(embedded))
            for conv in self.convs
        ]
        # conved_n [batch_size, n_filters, (seq_len - filter_sizes[n] + 1)]
        pooled = [
            F.max_pool1d(conv, conv.shape[2]).squeeze(2)
            for conv in conved
        ]
        # pooled_n [batch_size, n_filters]

        # cat
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat [batch_size, n_filters * len(filter_sizes)]
        out = self.fc(cat)
        # out [batch_size, output_dim]
        return out


class TextCNN2d(BaseModel):

    def __init__(self, n_filters, filter_sizes, output_dim, dropout, word_embedding, freeze):
        super().__init__()

        self.embedding_dim = word_embedding.vectors.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding.vectors), freeze=freeze)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, self.embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, _, text_lengths):
        # text [batch_size, seq_len]
        embedded = self.embedding(text).float()
        # embedded [batch_size, seq_len, emb_dim]
        embedded = embedded.unsqueeze(1)
        # embedded [batch_size, 1, seq_len, emb_dim]

        conved = [
            F.relu(conv(embedded)).squeeze(3)
            for conv in self.convs
        ]
        # conved_n [batch_size, n_filters, (seq_len - filter_sizes[n] + 1)]
        pooled = [
            F.max_pool1d(conv, conv.shape[2]).squeeze(2)
            for conv in conved
        ]
        # pooled_n [batch_size, n_filters]

        # cat
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat [batch_size, n_filters * len(filter_sizes)]
        out = self.fc(cat)
        # out [batch_size, output_dim]
        return out


# --------------------------------------- #


class RNN(BaseModel):

    def __init__(self, rnn_type, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, word_embedding, freeze, batch_first=True):
        super().__init__()

        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = word_embedding.vectors.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding.vectors), freeze=freeze)

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_dim,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.fc = nn.Linear(hidden_dim * n_layers, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, text, _, text_lengths):
        # sort
        text, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(text, text_lengths)
        # text [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text)).float()
        # embedded [batch_size, seq_len, emb_dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_seq_lengths, batch_first=self.batch_first)
        # packed_embedded [batch_size, seq_len, emb_dim]
        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # output [seq_len, batch_size, hidden_dim * num_direction]
        # hidden [n_layers * num_direction, batch_size, hidden_dim]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        # output [batch_size, seq_len, hidden_dim * num_direction]
        output = output[desorted_indices]

        # operate hidden
        hidden = torch.reshape(hidden, [output.shape[0], -1, output.shape[2]])
        # hidden [batch_size, batch_size(-1), hidden_dim * num_direction]
        hidden = torch.mean(hidden, dim=1)
        # hidden [batch_size, hidden_dim * num_direction]

        # operate output
        # output [batch_size, seq_len, hidden_dim * num_direction]
        output = torch.mean(output, dim=1)
        # output [batch_size, hidden_dim * num_direction]

        # add
        fc_input = self.dropout(output + hidden)
        # fc_input [batch_size, hidden_dim * num_direction]
        out = self.fc(fc_input)
        # out [batch_size, output_dim]
        return out


class RNNAttention(BaseModel):

    def __init__(self, rnn_type, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, word_embedding, freeze, batch_first=True):
        super().__init__()

        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = word_embedding.vectors.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding.vectors), freeze=freeze)

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_dim,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        if bidirectional:
            self.w = nn.Parameter(torch.randn(hidden_dim * 2), requires_grad=True)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.w = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.tanh = nn.Tanh()

    def forward(self, text, _, text_lengths):
        # sort
        text, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(text, text_lengths)
        # text [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text)).to(torch.float32)
        # embedded [batch_size, seq_len, emb_dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_seq_lengths, batch_first=self.batch_first)
        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # output [seq_len, batch_size, hidden_dim * num_direction]
        # hidden [n_layers * num_direction, batch_size, hidden_dim]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        # output [batch_size, seq_len, hidden_dim * num_direction]
        output = output[desorted_indices]

        # attention
        m = self.tanh(output)
        # m [batch_size, seq_len, hidden_dim * num_direction]

        # w [hidden_dim * 1(or 2)]
        score = torch.matmul(m, self.w)
        # score [batch_size, seq_len]

        alpha = F.softmax(score, dim=0).unsqueeze(-1)
        # alpha [batch_size, seq_len, 1]

        output_attention = output * alpha
        # output_attention [batch_size, seq_len, hidden_dim * num_direction]

        # operate hidden
        hidden = torch.reshape(hidden, [output.shape[0], -1, output.shape[2]])
        hidden = torch.mean(hidden, dim=1)
        # hidden [batch_size, hidden_dim * num_direction]

        # operate output_attention
        output_attention = torch.sum(output_attention, dim=1)
        # output_attention [batch_size, hidden_dim * num_direction]

        # operate output
        output = torch.sum(output, dim=1)
        # output [batch_size, hidden_dim * num_direction]

        # add
        fc_input = self.dropout(output + output_attention + hidden)
        # fc_input [batch_size, hidden_dim * num_direction]
        out = self.fc(fc_input)
        # out [batch_size, output_dim]
        return out


class TextRCNN(BaseModel):

    def __init__(self, rnn_type, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, word_embedding, freeze, batch_first=True):
        super().__init__()

        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_size = word_embedding.vectors.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding.vectors), freeze=freeze)

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.tanh = nn.Tanh()
        self.fc_cat = nn.Linear(hidden_dim * n_layers + self.embedding_size, self.embedding_size)
        self.fc = nn.Linear(self.embedding_size, output_dim)

    def forward(self, text, _, text_lengths):
        # sort
        text, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(text, text_lengths)
        # text [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text)).to(torch.float32)
        # embedded [batch_size, seq_len, emb_dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_seq_lengths, batch_first=self.batch_first)
        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # packed_output [seq_len, batch_size, hidden_dim * num_direction]
        # hidden [n_layers * num_direction, batch_size, hidden_dim]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        # output [batch_size, seq_len, hidden_dim * num_direction]
        output = output[desorted_indices]

        _, max_seq_len, _ = output.shape
        output = torch.cat((output, embedded), dim=2)
        # output [batch_size, seq_len, hidden_dim * num_direction + emb_dim]
        output = self.tanh(self.fc_cat(output))
        # output [batch_size, seq_len, emb_dim]
        output = torch.transpose(output, 1, 2)
        # output [batch_size, emb_dim, seq_len]
        output = F.max_pool1d(output, max_seq_len).squeeze().contiguous()
        # output [batch_size, emb_dim]
        output = self.fc(output)
        # output [batch_size, output_dim]
        return output


# --------------------------------------- #


class Bert(BaseModel):

    def __init__(self, bert_path, num_classes, word_embedding, trained=True):
        super(Bert, self).__init__()

        self.bert = BertModel.from_pretrained(bert_path)

        for param in self.bert.parameters():
            param.requires_grad = trained

        self.fc = nn.Linear(self.bert.config.to_dict()['hidden_size'], num_classes)

    def forward(self, context, bert_masks, seq_lens):
        # context [batch_size, sen_len]
        _, cls = self.bert(context, attention_mask=bert_masks)
        # cls [batch_size, H=768]

        out = self.fc(cls)
        # out [hidden_size, num_classes]
        return out


class BertCNN(nn.Module):

    def __init__(self, bert_path, num_filters, hidden_size, filter_sizes, dropout, num_classes, word_embedding,
                 trained=True):
        super(BertCNN, self).__init__()

        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = trained

        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=num_filters,
                kernel_size=(k, hidden_size))
            for k in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc_cnn = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        conved = F.relu(conv(x)).squeeze(3)
        pooled = F.max_pool1d(conved, conved.size(2)).squeeze(2)
        return pooled

    def forward(self, context, bert_masks, seq_len):
        # context [batch_size, sen_len]
        encoder_out, _ = self.bert(context, attention_mask=bert_masks)
        # encoder_out [batch_size, sen_len, H=768]

        # cnn
        encoder_out = self.dropout(encoder_out)
        out = encoder_out.unsqueeze(1)
        # out [batch_size, 1, sen_len, H=768]
        out = torch.cat([
            self.conv_and_pool(out, conv)
            for conv in self.convs
        ], dim=1)
        # out == [batch_size, n_filters * len(filter_sizes)]

        out = self.dropout(out)
        out = self.fc_cnn(out)
        # out [batch_size, num_classes]
        return out


class BertRNNAttention(nn.Module):

    def __init__(self, rnn_type, bert_path, hidden_dim, n_layers, bidirectional, batch_first, word_embedding,
                 dropout, num_classes, trained):
        super(BertRNNAttention, self).__init__()

        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = trained
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.bert.config.to_dict()['hidden_size'],
                               hidden_size=hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.bert.config.to_dict()['hidden_size'],
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.bert.config.to_dict()['hidden_size'],
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.w = nn.Parameter(torch.randn(hidden_dim * 2), requires_grad=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, text, bert_masks, seq_lens):
        # text [batch_size, seq_len]
        bert_sentence, bert_cls = self.bert(text, attention_mask=bert_masks)
        # bert_sentence [batch_size, sen_len, H=768]
        # bert_cls [batch_size, H=768]

        # rnn
        bert_sentence, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(bert_sentence, seq_lens)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(bert_sentence, sorted_seq_lengths,
                                                            batch_first=self.batch_first)
        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # output = [seq_len, batch_size, hidden_dim * bidirectional]
        # hidden [n_layers * num_direction, batch_size, hidden_dim]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        # output = [batch_size, seq_len, hidden_dim * bidirectional]
        output = output[desorted_indices]

        # attention
        # operate output_attention
        m = self.tanh(output)
        # m [batch_size, seq_len, hidden_dim * bidirectional]
        score = torch.matmul(m, self.w)
        # score [batch_size, seq_len]
        alpha = F.softmax(score, dim=0).unsqueeze(-1)
        # alpha [batch_size, seq_len, 1]
        output_attention = output * alpha
        # output_attention [batch_size, seq_len, hidden_dim * bidirectional]
        output_attention = torch.sum(output_attention, dim=1)
        # output_attention [batch_size, hidden_dim * bidirectional]
        # operate hidden
        hidden = torch.reshape(hidden, [output.shape[0], -1, output.shape[2]])
        hidden = torch.mean(hidden, dim=1)
        # hidden [batch_size, hidden_dim * bidirectional]
        # operate output
        output = torch.sum(output, dim=1)
        # output [batch_size, hidden_dim * bidirectional]
        # add
        fc_input = self.dropout(output + output_attention + hidden)
        # fc_input [batch_size, hidden_dim * bidirectional]

        out = self.fc(fc_input)
        # out [batch_size, num_classes]
        return out


class BertRCNN(BaseModel):

    def __init__(self, rnn_type, bert_path, hidden_dim, n_layers,
                 bidirectional, dropout, num_classes, word_embedding,
                 trained, batch_first=True):
        super().__init__()

        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = trained
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.bert.config.to_dict()['hidden_size'],
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.bert.config.to_dict()['hidden_size'],
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.bert.config.to_dict()['hidden_size'],
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fc = nn.Linear(hidden_dim * n_layers, num_classes)

    def forward(self, text, bert_masks, seq_lens):
        # text [batch_size, seq_len]
        bert_sentence, bert_cls = self.bert(text, attention_mask=bert_masks)
        # bert_sentence [batch_size, sen_len, H=768]
        # bert_cls [batch_size, H=768]

        # operate bert_cls
        bert_cls = bert_cls.unsqueeze(dim=1)
        # bert_cls [batch_size]
        bert_cls = bert_cls.repeat(1, bert_sentence.shape[1], 1)
        # bert_cls [batch_size, sen_len, 1]

        # operate bert_sentence
        bert_sentence = bert_sentence + bert_cls
        # bert_sentence [batch_size, sen_len, H=768]

        # rnn
        bert_sentence, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(bert_sentence, seq_lens)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(bert_sentence, sorted_seq_lengths,
                                                            batch_first=self.batch_first)
        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # output [seq_len, batch_size, hidden_dim * bidirectional]
        # hidden [n_layers * num_direction, batch_size, hidden_dim]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        # output [batch_size, seq_len, hidden_dim * bidirectional]
        output = output[desorted_indices]
        output = output.relu()

        # cnn
        _, max_seq_len, _ = output.shape
        out = torch.transpose(output, 1, 2)
        # output [batch_size, hidden_dim * bidirectional, seq_len]
        out = F.max_pool1d(out, max_seq_len).squeeze()
        # out [batch_size, hidden_dim * bidirectional]

        out = self.fc(out)
        # output [batch_size, output_dim]
        return out
