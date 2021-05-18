import torch.nn as nn
import torch.nn.functional as F
import torch
from base.base_model import BaseModel
from utils.model_utils import prepare_pack_padded_sequence
from transformers import BertModel


class MLP(BaseModel):
    """
    MLP
    """

    def __init__(self, hidden_dim, output_dim, dropout, word_embedding, freeze,
                 needed_by_mlp_num_hidden_layers, needed_by_mlp_max_seq_len):
        """
        1、data_process.py，文本提前做了padding
        2、model.py，可选隐藏层数
        """

        super().__init__()

        self.embedding_dim = word_embedding.vectors.shape[1]
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding.vectors), freeze=freeze)

        # 隐藏层数
        self.num_hidden_layers = needed_by_mlp_num_hidden_layers
        # 最大文本次长度
        self.max_seq_len = needed_by_mlp_max_seq_len
        # 对文本padding部分做全零初始化
        # self.embedding.weight.data[1]，即stoi['PAD']
        self.embedding.weight.data[1] = torch.zeros(self.embedding_dim)

        """
        普通循环方式1
        
        # 直接一、二、六层吧，默认一层
        if self.num_hidden_layers == 6:
            self.mlp = nn.Sequential(
                nn.Linear(self.max_seq_len * self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        elif self.num_hidden_layers == 2:
            self.mlp = nn.Sequential(
                nn.Linear(self.max_seq_len * self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.max_seq_len * self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
            
            
        普通循环方式2
        
        if self.num_hidden_layers == 6:
            self.mlp = nn.Sequential(
                nn.Linear(self.max_seq_len * self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                self.one_hidden_layer,
                self.one_hidden_layer,
                self.one_hidden_layer,
                self.one_hidden_layer,
                self.one_hidden_layer,
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        elif self.num_hidden_layers == 2:
            self.mlp = nn.Sequential(
                nn.Linear(self.max_seq_len * self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                self.one_hidden_layer,
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.max_seq_len * self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        """

        # 一个中间隐藏层
        self.one_hidden_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 所有中间隐藏层 - 1
        self.all_hidden_layer = nn.Sequential(self.one_hidden_layer)
        # 所有中间隐藏层 - 2
        for i in range(self.num_hidden_layers - 2):
            self.all_hidden_layer.add_module(str(i + 1), self.one_hidden_layer)

        # 隐藏层1层/n层
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

        # [batch_size, output_dim]

        return out


class TextCNN1d(BaseModel):
    """
    TextCNN(1d)

    class TextCNN(nn.Module):
    """

    def __init__(self, n_filters, filter_sizes, output_dim, dropout, word_embedding, freeze):
        super().__init__()

        self.embedding_dim = word_embedding.vectors.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding.vectors), freeze=freeze)

        # 卷积核list
        """
        1d的，
        in_channels=self.embedding_dim，词向量维度个的特征
        out_channels，输出是n_filters（因为有n个卷积核在做卷积，有n个feature-map）
        kernel_size，卷积核大小（只有横向的定义的窗口大小，没有纵向）        
        """
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, _, text_lengths):
        # text == [batch_size, seq_len]

        embedded = self.embedding(text).float()

        # embedded == [batch_size, seq_len, emb_dim]

        embedded = embedded.permute(0, 2, 1)

        # embedded == [batch_size, emb_dim, seq_len]

        # 卷积（多个卷积核）
        """
        embedded，做每个conv，并激活
        
        变化：
        [batch_size,    emb_dim,        seq_len                         ]
        [batch_size,    n_filters,      (seq_len - filter_sizes[n] + 1) ]
        """
        conved = [
            F.relu(conv(embedded))
            for conv in self.convs
        ]

        # conved_n == [batch_size, n_filters, (seq_len - filter_sizes[n] + 1)]

        # 池化（最大池化）
        """
        conv，
        将conv.shape[2]（也就是卷积核窗口大小）做max_pool1d，
        压扁为1（现在n_filters维的数就是池化后的值），
        再去掉
        """
        pooled = [
            F.max_pool1d(conv, conv.shape[2]).squeeze(2)
            for conv in conved
        ]

        # pooled_n == [batch_size, n_filters]

        # 拼接所有池化层，然后dropout
        """
        第2维变成了一张（n个卷积核*一个卷积核大小）的平面
        """
        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat == [batch_size, n_filters * len(filter_sizes)]

        # 线性映射，len(filter_sizes) * n_filters到output_dim
        out = self.fc(cat)

        # out == [batch_size, output_dim]

        return out


class TextCNN2d(BaseModel):
    """
    TextCNN(2d)

    class TextCNN(nn.Module):
    """

    def __init__(self, n_filters, filter_sizes, output_dim, dropout, word_embedding, freeze):
        """

        :param n_filters: 窗口的个数（卷积核的个数）
        :param filter_sizes: 窗口的大小
        :param output_dim: 输出层维度
        :param dropout:
        :param word_embedding: 预训练的词向量
        :param freeze: 冻结部分参数学习
        """

        super().__init__()

        # 词嵌入维度（embedding_dim）
        self.embedding_dim = word_embedding.vectors.shape[1]

        # 词嵌入
        """
        freeze：
        深度学习里，往往会用freeze部分参数的技巧，改变收敛位置，提高调参效率。
        实际使用的时候，往往在训练后期，只freeze浅层的参数，即靠近输入端的若干层。
        
        方法就是，反向传播到被freeze的那层就停止传播了。
        这样做可以提升效率，而且有缩小解空间的性质。
        至于正确性的话，因为训练后期，浅层的权值梯度接近全零向量，没有太大的更新意义。
        实在放心不下，可以每训练100轮解除freeze一下，然后完整反向传播、调参一次之后再freeze起来。

        nn.Embedding.from_pretrained():
                Creates Embedding instance from given 2-dimensional FloatTensor.
        freeze (boolean, optional): 
                If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``
        """
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding.vectors), freeze=freeze)

        # 卷积核list（过滤器）
        """
        拿到多个不同窗口大小的，2d卷积核，多个conv做并联（不是串联）
        可以有不同的卷积核设计，这个是比较常用的
        in_channels，1是因为是一维文本
        out_channels，输出是n_filters（因为有n个卷积核在做卷积，有n个feature-map）
        kernel_size，卷积核大小（自然是：横向的定义的窗口大小，纵向的词嵌入维度）        
        """
        """
        channels 分为三种：
        最初输入的图片样本的 channels ，取决于图片类型，
        （比如RGB，channels=3）。
        卷积核中的 in_channels ，就是要操作的图像数据的feather map张数，也就是卷积核的深度，
        （如果是第1次做卷积，就是样本图片的 channels；如果是第n次，是上一次卷积的 out_channels ）。
        卷积操作完成后输出的 out_channels ，取决于卷积核的数量，也是下层将产生的feather map数量，
        （此时的 out_channels 也会作为下一次卷积时的卷积核的 in_channels）。

        tip：
        上一层每个feature map，跟每个卷积核做卷积，都会产生下一层的一个feature map。
        有N个卷积核，下层就会产生N个feather map。

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel

        Shape:
            - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
            - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
        """
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, self.embedding_dim))
            for fs in filter_sizes
        ])

        # 线性映射，（len(filter_sizes) * n_filters）（卷积核大小*卷积核数量）到output_dim
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, _, text_lengths):
        # text == [batch_size, seq_len]

        # 嵌入
        embedded = self.embedding(text).float()

        # embedded == [batch_size, seq_len, emb_dim]

        # 留出第2维
        embedded = embedded.unsqueeze(1)

        # embedded == [batch_size, 1, seq_len, emb_dim]

        # 卷积（多个卷积核）
        """
        embedded，做每个conv，并激活，去掉第4维

        变化：
        [batch_size,    1,              seq_len,                            emb_dim ]
        [batch_size,    n_filters,      (seq_len - filter_sizes[n] + 1),    1       ]
        [batch_size,    n_filters,      (seq_len - filter_sizes[n] + 1)             ]
        （batch_size，输入维（1），隐藏维（第几个字），权重维（词嵌入））
        （batch_size，输出维（n层map feature），隐藏维（1层map feature的长或宽；词个数-窗口大小+1），权重维（融入进了前两个维度，压扁了，变为1））
        （batch_size，输入维，隐藏维）
        （只能说，pytorch实现的参数，就是定义的这样输入和输出）

        Shape:
            - Input: :math:`(N, C_{in}, H_{in}, W_{in})`，
            - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
        """
        conved = [
            F.relu(conv(embedded)).squeeze(3)
            for conv in self.convs
        ]

        # conved_n == [batch_size, n_filters, (seq_len - filter_sizes[n] + 1)]

        # 池化（最大池化）
        """
        conv，
        将conv.shape[2]（也就是卷积核窗口大小）做max_pool1d，
        压扁为1（现在n_filters维的数就是池化后的值），
        再去掉
        
        torch.nn.functional.max_pool1d(input, kernel_size)
        """

        pooled = [
            F.max_pool1d(conv, conv.shape[2]).squeeze(2)
            for conv in conved
        ]

        # pooled_n == [batch_size, n_filters]

        # 拼接所有池化层，然后dropout
        """
        第2维变成了一张（n个卷积核*一个卷积核大小）的平面
        torch.cat()，在给定维度上对输入的张量序列进行连接，        
        """
        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat == [batch_size, n_filters * len(filter_sizes)]

        # 线性映射，len(filter_sizes) * n_filters到output_dim
        out = self.fc(cat)

        # out == [batch_size, output_dim]

        return out


# --------------------------------------- #


class RNN(BaseModel):
    """
    RNN/LSTM/GRU

    rnn实现参考，D:/SoftwareDevelopmentKit/Anaconda3/Lib/site-packages/torch/nn/modules/rnn.py
    """

    def __init__(self, rnn_type, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, word_embedding, freeze, batch_first=True):
        """

        :param rnn_type: rnn类型
        :param hidden_dim: 隐藏层维度
        :param output_dim: 输出层维度
        :param n_layers: n层
        :param bidirectional: 是否双向
        :param dropout: 除最后一层，每一层的输出都进行dropout，默认为: 0
        :param word_embedding: 预训练的词向量
        :param freeze: 冻结
        :param batch_first: 【这里为True！！】，输入输出的数据格式为 [batch, seq, feature]
        """

        super().__init__()

        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding_dim = word_embedding.vectors.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding.vectors), freeze=freeze)

        """
        注意，
        以下三个RNN，都是默认batch_first=Fasle的，
        这里使用True
        batch_first: 
            If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``
        """
        # self.rnn，确定rnn类型
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

        # 线性映射，hidden_dim * n_layers（隐藏层维度*n层，比如双向最后是拼接）到output_dim
        self.fc = nn.Linear(hidden_dim * n_layers, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, text, _, text_lengths):
        """

        :param text: 输入的句子，
        第一版，batch_first配置在config里面，为False
        第二版，batch_first配置在这个类缺省参数里面，为True
        现在它全是正常的顺序[batch_size, seq_len]，
        CNN第二版不用这个参数
        :param text_lengths: 句子长度
        """

        # 降序文本，降序文本长度，一开始不排序的索引
        text, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(text, text_lengths)
        # text [batch_size, seq_len]

        # 先embedding，然后dropout
        embedded = self.dropout(self.embedding(text)).float()
        # embedded [batch_size, seq_len, emb_dim]

        # pack
        """
        pack_padded_sequence()，Packs a Tensor containing padded sequences of variable length，
        pack一个（已经padding过的）sequence，记下了做了padding的长度的list。

        一般在处理数据时就已经将序列pad成等长了，但是LSTM需要一种方法来告知自己处理变长输入，
        一个batch里的序列不一定等长，需要pad操作，用0把它们都填充成max_length长度。

        LSTM的一次forward对应一个time step，接收的是across batches的输入，
        这就导致短序列可能在当前time step上已经结束，而你还是在给它输入东西（pad），
        这就会对结果产生影响（可以对照公式看看，即便输入全0还是会有影响），
        我们想要的效果是，LSTM知道batch中每个序列的长度，等到某个序列输入结束后下面的time step就不带它了。

        batch_first=self.batch_first，如果要保存batch_first的维度。
        
        nn.utils.rnn.pack_padded_sequence():
            Packs a Tensor containing padded sequences of variable length.
    
            :attr:`input` can be of size ``T x B x *`` where `T` is the length of the
            longest sequence (equal to ``lengths[0]``), ``B`` is the batch size, and
            ``*`` is any number of dimensions (including 0). If ``batch_first`` is
            ``True``, ``B x T x *`` :attr:`input` is expected.
        """
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_seq_lengths, batch_first=self.batch_first)
        # packed_embedded [batch_size, seq_len, emb_dim]

        """
        注意，
        这里batch_first=True，
        会保护batch_size的维度，
        下面的情况是官方的False情况的理解，
        区别只是pytorch会自动先把两个维度调换，
        输出的维度是一样的
        """
        # rnn、gru
        if self.rnn_type in ['rnn', 'gru']:
            # rnn
            """
            最后一层隐藏层状态集合，所有层最后一时刻隐藏层状态
            packed_output，(seq_len, batch, num_directions * hidden_size)
            hidden，(num_layers * num_directions, batch, hidden_size)
            
            Outputs: output, h_n
                - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
                  containing the output features (`h_t`) from the last layer of the RNN,
                  for each `t`.  If a :class:`torch.nn.utils.rnn.PackedSequence` has
                  been given as the input, the output will also be a packed sequence.
                  For the unpacked case, the directions can be separated
                  using ``output.view(seq_len, batch, num_directions, hidden_size)``,
                  with forward and backward being direction `0` and `1` respectively.
                  Similarly, the directions can be separated in the packed case.

                - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
                  containing the hidden state for `t = seq_len`.
                  Like *output*, the layers can be separated using
                  ``h_n.view(num_layers, num_directions, batch, hidden_size)``.
            """

            # gru
            """
            最后一层隐藏层状态集合，所有层最后一时刻隐藏层状态
            packed_output，(seq_len, batch, num_directions * hidden_size)
            hidden，(num_layers * num_directions, batch, hidden_size)
            
            Outputs: output, h_n
                - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
                  containing the output features h_t from the last layer of the GRU,
                  for each `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
                  given as the input, the output will also be a packed sequence.
                  For the unpacked case, the directions can be separated
                  using ``output.view(seq_len, batch, num_directions, hidden_size)``,
                  with forward and backward being direction `0` and `1` respectively.
                  Similarly, the directions can be separated in the packed case.

                - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
                  containing the hidden state for `t = seq_len`
                  Like *output*, the layers can be separated using
                  ``h_n.view(num_layers, num_directions, batch, hidden_size)``.
            """

            packed_output, hidden = self.rnn(packed_embedded)

        # lstm
        else:
            """
            最后一层外部状态集合，（所有层最后一个时刻外部状态，所有层最后一个时刻内部状态）

            packed_output，(seq_len, batch, num_directions * hidden_size)
            hidden，(num_layers * num_directions, batch, hidden_size)
            cell，(num_layers * num_directions, batch, hidden_size)

            output，
            最后一层，每个time_step的输出h，
            num_directions * hidden_size可以看出来，
            （双向，每个time_step的输出h = [h正向, h逆向]，是同一个time_step的正向和逆向的h连接起来！）

            h_n，
            每一层，最后一个time_step的输出h，
            num_layers * num_directions可以看出来，
            （双向，单独保存前向和后向的最后一个time_step的输出h）
            （一层的话，h_n和output最后一个是一样的【应该一定】）

            c_n，
            和h_n一样意思，保存的是c
            
            
            Outputs: output, (h_n, c_n)
                - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
                  containing the output features `(h_t)` from the last layer of the LSTM,
                  for each `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
                  given as the input, the output will also be a packed sequence.
                  For the unpacked case, the directions can be separated
                  using ``output.view(seq_len, batch, num_directions, hidden_size)``,
                  with forward and backward being direction `0` and `1` respectively.
                  Similarly, the directions can be separated in the packed case.

                - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
                  containing the hidden state for `t = seq_len`.
                  Like *output*, the layers can be separated using
                  ``h_n.view(num_layers, num_directions, batch, hidden_size)`` and similarly for *c_n*.

                - **c_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
                  containing the cell state for `t = seq_len`.
            """
            packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # output（最后一层的每个词时刻），hidden（每一层的最后一时刻）
        # output [seq_len, batch_size, hidden_dim * num_direction]
        # hidden [n_layers * num_direction, batch_size, hidden_dim]

        # unpack
        """
        unpack一个（经过packed的）sequence，
        output，output对应pad长度
        pad_packed_sequence()，Pads a packed batch of variable length sequences.
        
        这里的batch_first，把batch_size调回第一维
        """
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        # output [batch_size, seq_len, hidden_dim * num_direction]

        # 文本顺序调回输入时的顺序
        output = output[desorted_indices]

        # 处理hidden
        # output [batch_size, seq_len, hidden_dim * num_direction]
        # hidden [n_layers * num_direction, batch_size, hidden_dim]

        # hidden要对应的维度，直接取自output的某一维度
        """
        -1，指第1维自动适应计算（不动），记作batch_size(-1)        
        """
        hidden = torch.reshape(hidden, [output.shape[0], -1, output.shape[2]])
        # hidden [batch_size, batch_size(-1), hidden_dim * num_direction]

        # 在第1维求均值，去掉第1维
        hidden = torch.mean(hidden, dim=1)
        # hidden [batch_size, hidden_dim * num_direction]

        # 处理output
        # output [batch_size, seq_len, hidden_dim * num_direction]

        # 在第1维求均值，去掉第1维
        output = torch.mean(output, dim=1)
        # output [batch_size, hidden_dim * num_direction]

        # 相加
        # output [batch_size, hidden_dim * num_direction]
        # hidden [batch_size, hidden_dim * num_direction]
        # output（最后一层的每个词时刻），hidden（每一层的最后一时刻），直接相加（不是拼接），再dropout
        fc_input = self.dropout(output + hidden)

        # 线性映射，hidden_dim * n_layers到output_dim
        out = self.fc(fc_input)
        # output [batch_size, output_dim]

        return out


class RNNAttention(BaseModel):
    """
    RNN/LSTM/GRU+Attention
    """

    def __init__(self, rnn_type, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, word_embedding, freeze, batch_first=True):
        """

        :param rnn_type:
        :param hidden_dim:
        :param output_dim:
        :param n_layers:
        :param bidirectional:
        :param dropout:
        :param word_embedding:
        :param freeze:
        :param batch_first:
        """

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

        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # 激活
        self.tanh = nn.Tanh()

        """
        注释掉的，原来想写加性模型的注意力机制
        self.u = nn.Parameter(torch.Tensor(self.hidden_dim * 2,self.hidden_dim*2))
        self.w = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)
        """

        # 初始化随机权重、线性映射
        """
        如果没有声明bidirectional，
        默认是True的（试过条件语句打印；但是没查，可能是随机数非0为true）
        """

        # 双向
        if bidirectional:
            # 维度hidden_dim * 2，
            # nn.Parameter(requires_grad=True)，要求梯度的参数
            self.w = nn.Parameter(torch.randn(hidden_dim * 2), requires_grad=True)
            # hidden_dim * 2到output_dim
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        # 单向
        else:
            # 维度hidden_dim
            self.w = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)
            # hidden_dim到output_dim
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, _, text_lengths):

        # text [batch_size, seq_len]
        text, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(text, text_lengths)

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

        # 注意力,
        """
        用的是中科院自动所论文提出的the attention for relation classification tasks
        """
        # output [batch_size, seq_len, hidden_dim * num_direction]
        # 计算m
        m = self.tanh(output)
        # m [batch_size, seq_len, hidden_dim * num_direction]

        # w [hidden_dim * 1(or 2)]
        # 计算score（注意力打分函数），矩阵乘
        """
        torch.matmul()，
        w一维，和m最后一维相同，相乘会消掉m最后一维
        """
        score = torch.matmul(m, self.w)
        # score [batch_size, seq_len]

        # 计算alpha（注意力分布）
        """
        dim=0表示针对文本中的每个词的输出softmax
        softmax(dim=0)，对每个数做softmax（第1维，废话）
        unsqueeze(-1)，在最后一维插入一个维度
        """
        alpha = F.softmax(score, dim=0).unsqueeze(-1)
        # alpha [batch_size, seq_len, 1]

        # output [batch_size, seq_len, hidden_dim * num_direction]
        # 计算r（加权平均），点乘
        output_attention = output * alpha
        # output_attention [batch_size, seq_len, hidden_dim * num_direction]

        # 处理hidden
        # output [batch_size, seq_len, hidden_dim * num_direction]
        # hidden [n_layers * num_direction, batch_size, hidden_dim]
        hidden = torch.reshape(hidden, [output.shape[0], -1, output.shape[2]])
        hidden = torch.mean(hidden, dim=1)
        # hidden [batch_size, hidden_dim * num_direction]

        # 处理output_attention
        # output_attention [batch_size, seq_len, hidden_dim * num_direction]
        # output_attention，第1维求和，消掉第1维
        output_attention = torch.sum(output_attention, dim=1)
        # output_attention [batch_size, hidden_dim * num_direction]

        # 处理output
        # output [batch_size, seq_len, hidden_dim * num_direction]
        output = torch.sum(output, dim=1)
        # output [batch_size, hidden_dim * num_direction]

        # 相加
        # output [batch_size, hidden_dim * num_direction]
        # output_attention [batch_size, hidden_dim * num_direction]
        # hidden [batch_size, hidden_dim * num_direction]
        # output，attention的output，hidden，三个相加，然后dropout
        fc_input = self.dropout(output + output_attention + hidden)
        # fc_input [batch_size, hidden_dim * num_direction]

        # 线性映射，hidden_dim * n_layers到output_dim
        out = self.fc(fc_input)
        # output [batch_size, output_dim]

        return out


class TextRCNN(BaseModel):
    """
    textrcnn
    """

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

        # 线性映射，hidden_dim * n_layers + self.embedding_size，到，self.embedding_size
        self.fc_cat = nn.Linear(hidden_dim * n_layers + self.embedding_size, self.embedding_size)
        # 线性映射，self.embedding_size到output_dim
        self.fc = nn.Linear(self.embedding_size, output_dim)

    def forward(self, text, _, text_lengths):
        # text [batch_size, seq_len]
        text, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(text, text_lengths)
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
        # output [batch_size, seq_len, hidden_dim * num_direction]
        # embedded [batch_size, seq_len, emb_dim]
        # 拼接，激活（在第3维，经过rnn+原始embedded）
        output = torch.cat((output, embedded), dim=2)
        # output [batch_size, seq_len, hidden_dim * num_direction + emb_dim]
        output = self.tanh(self.fc_cat(output))
        # output [batch_size, seq_len, emb_dim]
        output = torch.transpose(output, 1, 2)
        # output [batch_size, emb_dim, seq_len]
        # 池化，卷积核大小为max_seq_len，再去掉
        output = F.max_pool1d(output, max_seq_len).squeeze().contiguous()
        # output [batch_size, emb_dim]
        output = self.fc(output)
        # output [batch_size, output_dim]
        return output


# --------------------------------------- #


class Bert(BaseModel):
    """
    Bert
    """

    def __init__(self, bert_path, num_classes, word_embedding, trained=True):
        """

        :param bert_path: bert预训练模型路径
        :param num_classes: 分类数
        :param word_embedding: None
        :param trained: 是否训练bert参数
        """

        super(Bert, self).__init__()

        # 从bert预训练模型文件，加载BertModel
        self.bert = BertModel.from_pretrained(bert_path)

        # 是否对bert进行训练
        """
        即是否对可训练参数求梯度
        
        原注释————不对bert进行训练
        """
        for param in self.bert.parameters():
            param.requires_grad = trained

        # 线性映射，bert的隐藏层维度，到，分类数
        self.fc = nn.Linear(self.bert.config.to_dict()['hidden_size'], num_classes)

    def forward(self, context, bert_masks, seq_lens):
        """
        :param context: 输入的句子序列

        :param bert_masks:
        构造的attention可视域的attention_mask，
        mask掉该位置后面的部分是为了保证模型不提前知道正确输出，
        因为那是要预测的呀！

        :param seq_lens: 句子长度
        """

        # context [batch_size, sen_len]

        # context传入bert模型，bert_masks标识要预测的部分
        _, cls = self.bert(context, attention_mask=bert_masks)
        # _ [batch_size, sen_len, H=768]
        # cls [batch_size, H=768]

        # 直接用cls预测
        out = self.fc(cls)
        # cls [hidden_size, num_classes]

        return out


class BertCNN(nn.Module):
    """
    Bert+TextCNN2D
    """

    def __init__(self, bert_path, num_filters, hidden_size, filter_sizes, dropout, num_classes, word_embedding,
                 trained=True):
        """

        :param bert_path:
        :param num_filters:
        :param hidden_size:
        :param filter_sizes:
        :param dropout:
        :param num_classes:
        :param word_embedding:
        :param trained:
        """

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

        # 线性映射，（len(filter_sizes) * n_filters）（卷积核大小*卷积核数量）到num_classes
        self.fc_cnn = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        """
        封装卷积和池化
        """

        conved = F.relu(conv(x)).squeeze(3)
        pooled = F.max_pool1d(conved, conved.size(2)).squeeze(2)

        return pooled

    def forward(self, context, bert_masks, seq_len):
        # context [batch_size, sen_len]

        # 文本encode，文本cls
        encoder_out, _ = self.bert(context, attention_mask=bert_masks)
        # encoder_out [batch_size, sen_len, H=768]
        # _ [batch_size, H=768]

        # encoder_out做dropout
        encoder_out = self.dropout(encoder_out)

        out = encoder_out.unsqueeze(1)
        # out [batch_size, 1, sen_len, H=768]

        # 卷积池化，再拼接
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
    """
    Bert+Bi-RNN
    """

    def __init__(self, rnn_type, bert_path, hidden_dim, n_layers, bidirectional, batch_first, word_embedding,
                 dropout, num_classes, trained):
        """

        :param rnn_type:
        :param bert_path:
        :param hidden_dim:
        :param n_layers:
        :param bidirectional: 这里是True，配置写了
        :param batch_first: 这里是True，配置写了
        :param word_embedding:
        :param dropout:
        :param num_classes:
        :param trained:
        """

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
            # 输入维度为bert的隐藏层维度
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

        """
        配置写双向，直接用双向
        """
        self.w = nn.Parameter(torch.randn(hidden_dim * 2), requires_grad=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, text, bert_masks, seq_lens):
        # text [batch_size, seq_len]
        bert_sentence, bert_cls = self.bert(text, attention_mask=bert_masks)
        # bert_sentence [batch_size, sen_len, H=768]
        # cls [batch_size, H=768]
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

        # 注意力,
        m = self.tanh(output)
        score = torch.matmul(m, self.w)
        alpha = F.softmax(score, dim=0).unsqueeze(-1)
        output_attention = output * alpha
        output_attention = torch.sum(output_attention, dim=1)

        # hidden [n_layers * num_direction, batch_size, hidden_dim]
        hidden = torch.reshape(hidden, [output.shape[0], -1, output.shape[2]])
        hidden = torch.mean(hidden, dim=1)
        # hidden [batch_size, hidden_dim * bidirectional]
        output = torch.sum(output, dim=1)
        # output [batch_size, hidden_dim * bidirectional]
        fc_input = self.dropout(output + output_attention + hidden)
        # fc_input [batch_size, hidden_dim * bidirectional]
        out = self.fc(fc_input)
        # out [batch_size, num_classes]

        return out


class BertRCNN(BaseModel):
    """
    BertRCNN
    """

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

        """
        torch.randn(33, 55).repeat(2,1).size()
        --->>>torch.Size([66, 55])
        
        变化到倍数维度
        """
        bert_cls = bert_cls.unsqueeze(dim=1)
        # bert_cls [batch_size]

        bert_cls = bert_cls.repeat(1, bert_sentence.shape[1], 1)
        # bert_cls [batch_size, sen_len, 1]
        # bert_sentence [batch_size, sen_len, H=768]

        bert_sentence = bert_sentence + bert_cls
        # bert_sentence [batch_size, sen_len, H=768]

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
        output = output.relu()
        # output = [batch_size, seq_len, hidden_dim * bidirectional]
        _, max_seq_len, _ = output.shape
        out = torch.transpose(output, 1, 2)
        # output = [batch_size, hidden_dim * bidirectional, seq_len]

        out = F.max_pool1d(out, max_seq_len).squeeze()

        # output = [batch_size, hidden_dim * bidirectional]

        out = self.fc(out)

        # output = [batch_size, output_dim]

        return out
