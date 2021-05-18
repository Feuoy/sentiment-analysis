# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss


# 1、gelu激活函数
def gelu(x):
    """
    在BERT使用的是GELU激活函数，高斯误差线性单元，

    https://github.com/huggingface/transformers/blob/master/src/transformers/activations.py，
    torch.__version__ < "1.4.0"，的gelu激活函数实现；不然，可以用torch.nn.functional.gelu

    Original Implementation of the gelu activation function in Google Bert repo when initially created.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    This is now written in C in torch.nn.functional
    Also see https://arxiv.org/abs/1606.08415
    """

    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# 2、激活函数字典
ACT2FN = {
    # "gelu": torch.nn.functional.gelu,
    "gelu": gelu,
    "relu": torch.nn.functional.relu
}


# 0、BertModel的配置类（called by 14、初始化模型参数）
class BertConfig(object):
    """
    Configuration class to store the configuration of a `BertModel`.

    Args:
            vocab_size (:obj:`int`, optional, defaults to 30522):
                Vocabulary size of the BERT model. Defines the different tokens that
                can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.BertModel`.
            hidden_size (:obj:`int`, optional, defaults to 768):
                Dimensionality of the encoder layers and the pooler layer.
            num_hidden_layers (:obj:`int`, optional, defaults to 12):
                Number of hidden layers in the Transformer encoder.
            num_attention_heads (:obj:`int`, optional, defaults to 12):
                Number of attention heads for each attention layer in the Transformer encoder.
            intermediate_size (:obj:`int`, optional, defaults to 3072):
                Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
            hidden_act (:obj:`str` or :obj:`function`, optional, defaults to "gelu"):
                The non-linear activation function (function or string) in the encoder and pooler.
                If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob (:obj:`float`, optional, defaults to 0.1):
                The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob (:obj:`float`, optional, defaults to 0.1):
                The dropout ratio for the attention probabilities.
            max_position_embeddings (:obj:`int`, optional, defaults to 512):
                The maximum sequence length that this model might ever be used with.
                Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
            type_vocab_size (:obj:`int`, optional, defaults to 2):
                The vocabulary size of the `token_type_ids` passed into :class:`~transformers.BertModel`.
            initializer_range (:obj:`float`, optional, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
                The epsilon used by the layer normalization layers.

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        **kwargs
    ):
    """

    def __init__(self,
                 vocab_size,  # 字典字数
                 hidden_size=384,  # 隐藏层维度，也就是字向量维度
                 num_hidden_layers=6,  # transformer block 的个数
                 num_attention_heads=12,  # 注意力机制"头"的个数
                 intermediate_size=384 * 4,  # encoder的中间隐层（例如feedforward层）的神经元数（线性映射的维度）
                 hidden_act="gelu",  # 激活函数
                 hidden_dropout_prob=0.4,  # 全连接层dropout的概率
                 attention_probs_dropout_prob=0.4,  # 注意力层dropout的概率
                 max_position_embeddings=512 * 2,  # 最大位置编码
                 type_vocab_size=256,  # tokens的类型数，预留了256个分类；用到的只有0和1，即上下句类数，即是Segment A和 Segment B
                 initializer_range=0.02  # 用来初始化模型参数的标准差
                 ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


# 1、embedding，字嵌入、位置嵌入和tokens_type，构造tokens嵌入
class BertEmbeddings(nn.Module):
    """
    embedding，字嵌入、位置嵌入和tokens_type，构造tokens嵌入，

    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        # 继承自nn.Module，父类初始化
        super(BertEmbeddings, self).__init__()

        # 字嵌入，vocab_size嵌入为hidden_size
        # padding填0
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)

        # tokens类型嵌入，type_vocab_size嵌入为hidden_size，
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 两个embedding矩阵的权重，权重初始化
        # torch.nn.init.orthogonal_(tensor)，正交初始化
        nn.init.orthogonal_(self.word_embeddings.weight)
        nn.init.orthogonal_(self.token_type_embeddings.weight)

        # 两个embedding矩阵的权重，归一化
        # torch.norm()，返回给定矩阵的范数；p=2，取范数（模）为2；dim=1，在第二维计算（hidden_size）
        epsilon = 1e-8
        self.word_embeddings.weight.data = self.word_embeddings.weight.data.div(
            torch.norm(self.word_embeddings.weight, p=2, dim=1, keepdim=True).data + epsilon
        )
        self.token_type_embeddings.weight.data = self.token_type_embeddings.weight.data.div(
            torch.norm(self.token_type_embeddings.weight, p=2, dim=1, keepdim=True).data + epsilon
        )

        # self.LayerNorm，层归一化
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # 源码用的是
        # self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)

        # self.dropout实例属性
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, positional_enc, token_type_ids=None):
        """
        forward，前向传播

        :param input_ids: 输入文本 [batch_size(句数), sequence_length(句子长度)]
        :param positional_enc: 位置编码 [sequence_length, embedding_dimension]
        :param token_type_ids: BERT训练的时候, 第一句是0, 第二句是1，维度应该是[sequence_length]
        :return: tokens嵌入[batch_size, sequence_length, embedding_dimension]
        """

        # 字嵌入
        # self.word_embeddings()，即nn.Embedding()，会自己实现，字向量化，
        # 把input_ids[batch_size, sequence_length]，通过vocab_size，
        # 嵌入为words_embeddings[batch_size, sequence_length, hidden_size](真的)，
        words_embeddings = self.word_embeddings(input_ids)
        # print("字嵌入：")
        # print(words_embeddings.size())

        # tokens类型嵌入
        # SegmentEmbedding，句子划分的信息，这里就前句和后句，0和1
        # 若 token_type_ids 为 none，有赋值为2，一般不会为None
        if token_type_ids is None:
            # 返回与input_ids相同维度的全0矩阵
            token_type_ids = torch.zeros_like(input_ids)

        # self.token_type_embeddings()，即nn.Embedding()，会自己实现，嵌入，
        # 把token_type_ids[sequence_length]，通过token_type_ids，
        # 嵌入为token_type_embeddings[batch_size, sequence_length, hidden_size](真的)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # print("tokens类型嵌入：")
        # print(token_type_embeddings.size())

        # 位置嵌入
        # print("位置嵌入：")
        # print(positional_enc.size())

        # tokens嵌入[batch_size, sequence_length, embedding_dimension]
        # 单词本身的向量，单词所在句子中位置的向量，句子所在单个训练文本中位置的向量
        # words_embeddings      [batch_size,    sequence_length,  hidden_size]，
        # positional_enc        [batch_size,    sequence_length,  embedding_dimension]，
        # token_type_embeddings [batch_size,    sequence_length,  hidden_size]，
        # 三者合在一起的tensor，表明一个word在三种维度下的特定含义
        # 三者的维度是一样的，打印出是一样的
        embeddings = words_embeddings + positional_enc + token_type_embeddings

        # 归一化
        embeddings = self.LayerNorm(embeddings)
        # dropout
        embeddings = self.dropout(embeddings)

        return embeddings


# 2、LayerNorm，层归一化（util）
class BertLayerNorm(nn.Module):
    """
    LayerNorm，层归一化

    源码用的是BertLayerNorm = torch.nn.LayerNorm
    """

    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """

        super(BertLayerNorm, self).__init__()

        # 权重，实例属性
        # 返回一个全为1的Tensor，维度hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 偏置，实例属性
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        # eps
        self.variance_epsilon = eps

    def forward(self, x):
        # LayerNorm公式
        # 以行求均值
        u = x.mean(-1, keepdim=True)
        # 以行求方差
        s = (x - u).pow(2).mean(-1, keepdim=True)
        # 每一行的每个元素，减去这行的均值，再除以这行的方差，得到归一化后的数值
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


# 3、SelfAttention，自注意力机制层
class BertSelfAttention(nn.Module):
    """
    自注意力机制层
    """

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()

        # 判断embedding_dimension是否可以被num_attention_heads整除
        if config.hidden_size % config.num_attention_heads != 0:
            # 不可以的话抛出异常
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        # 头数量
        self.num_attention_heads = config.num_attention_heads
        # 一份头的长度
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 所有头分长度
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 线性映射，得到Q, K, V
        # 查询向量，键向量，值向量
        # 从hidden_size，映射到all_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 配置dropout的概率
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """
        做变形

        :param x: QKV中的一个，[batch_size, seq_length, all_head_size]
        :return:  QKV中的一个，[batch_size, num_attention_heads, seq_length, attention_head_size]
        """

        # reshape，变换形状
        # 要变成的维度，<x的前两个维度，头数量，头尺寸>
        # x.size()[:-1]，是torch.Szie
        # (self.num_attention_heads, self.attention_head_size)，是tuple，不是列表生成式
        # new_x_shape，是torch.Szie
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)

        # x，维度[batch_size, seq_length, num_attention_heads, attention_head_size]
        # *new_x_shape，将new_x_shape拆分成向内一级的元素，这里然后作为参数
        # view()，按照*new_x_shape的序列，作为维度变形
        x = x.view(*new_x_shape)

        # x，维度[batch_size, num_attention_heads, seq_length, attention_head_size]
        # tensor.permute()，按照原来维度次序为(0, 1, 2, 3)变形为(0, 2, 1, 3)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, get_attention_matrices=False):
        """

        :param hidden_states: 上一个操作完后的隐藏层矩阵
        :param attention_mask:
        可选，构造的attention可视域的attention_mask，mask掉该位置后面的部分是为了保证模型不提前知道正确输出，因为那是要预测的呀！
        an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
        selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
        input sequence length in the current batch. It's the mask that we typically use for attention when
        a batch has varying length sentences.
        :param get_attention_matrices: 是否输出注意力矩阵, 可用于可视化
        :return: context_layer, attention_probs_ 或 context_layer, None
        """

        # hidden_states做线性映射，得到Q, K, V
        # Q, K, V此时的维度为[batch_size, seq_length, embedding_dim]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # 把Q，K，V分割成num_heads份
        # Q, K, V做变形，维度变为[batch_size, num_attention_heads, seq_length, attention_head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 做Scaled Dot Product Attention
        # K转置后两维，方便矩阵相乘，维度变为[batch_size, num_attention_heads, attention_head_size, seq_length]
        # Q与K（的转置）求点积（矩阵相乘），两个向量越相似，它们的点积就越大
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 除以K的dimension（即一个头的尺寸），开方以归一为标准正态分布
        # attention_scores: [batch_size, num_attention_heads, seq_length, seq_length]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # attention_mask[batch_size, 1, 1, seq_length]，是这个维度，
        # 元素相加后, 会广播到维度: attention_scores[batch_size, num_attention_heads, seq_length, seq_length]
        # 得到新的注意力分值
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # 在最后一个维度上做softmax归一化，使得每个字跟其它所有字的注意力权重的和为1
        # attention_probs_[batch_size, num_attention_heads, seq_length, seq_length]，同时也是注意力机制可视化的返回（这一层函数的）
        # Normalize the attention scores to probabilities.
        attention_probs_ = nn.Softmax(dim=-1)(attention_scores)

        # 做dropout
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs_)

        # 注意力矩阵（QK）[batch_size, num_attention_heads, seq_length, seq_length]，
        # 加权给V[batch_size, num_attention_heads, seq_length, attention_head_size]
        # context_layer对于原来的V的维度没有变化，[batch_size, num_attention_heads, seq_length, attention_head_size]
        context_layer = torch.matmul(attention_probs, value_layer)

        # 进行reshape
        # contiguous：
        # view只能用在contiguous的variable上。
        # 如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy。
        # [batch_size, num_attention_heads, seq_length, attention_head_size]
        # 转置，[batch_size, seq_length, num_attention_heads, attention_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # <batch_size, seq_length> + (all_head_size) 的torch.Size
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # 再按*new_context_layer_shape的形状，变形回来
        # 得到[batch_size, seq_length, embedding_dimension(即all_head_size)]
        context_layer = context_layer.view(*new_context_layer_shape)

        # 若get_attention_matrices为True
        if get_attention_matrices:
            # SelfAttention后的矩阵，attention矩阵用来可视化
            return context_layer, attention_probs_

        # SelfAttention后的矩阵
        return context_layer, None


# 4、封装的残差连接和LayerNorm, 用于处理SelfAttention的输出
class BertSelfOutput(nn.Module):
    """
    封装的残差连接和LayerNorm, 用于处理SelfAttention的输出
    """

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()

        # 铺底，线性映射一个全连接层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # LayerNorm
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # 线性映射
        hidden_states = self.dense(hidden_states)
        # dropout
        hidden_states = self.dropout(hidden_states)
        # input_tensor，tokens嵌入
        # hidden_states和input_tensor都是[batch_size, seq_length, embedding_dimension]
        # 残差连接后进行LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 5、封装：多头注意力机制部分, 残差连接和LayerNorm
class BertAttention(nn.Module):
    """
    封装：多头注意力机制部分, 残差连接和LayerNorm
    """

    def __init__(self, config):
        super(BertAttention, self).__init__()
        # 自注意力
        self.self = BertSelfAttention(config)
        # 残差连接和layerNorm
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, get_attention_matrices=False):
        # 自注意力
        self_output, attention_matrices = self.self(input_tensor, attention_mask,
                                                    get_attention_matrices=get_attention_matrices)
        # 残差连接和layerNorm
        attention_output = self.output(self_output, input_tensor)

        # 处理后的矩阵，attention_matrices
        return attention_output, attention_matrices


# 6、feedforward，前馈神经网络
class BertIntermediate(nn.Module):
    """
    feedforward，其实就是两层线性映射并用激活函数激活，比如ReLU，
    下一层线性映射在下面BertOutput()做，这里用的是Gelu
    """

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        # 全连接层，hidden_size线性映射到intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 激活函数
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        # 线性映射
        hidden_states = self.dense(hidden_states)
        # 激活
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 7、封装的LayerNorm和残差连接, 用于处理FeedForward层的输出
class BertOutput(nn.Module):
    """
    封装的LayerNorm和残差连接, 用于处理FeedForward层的输出
    """

    def __init__(self, config):
        super(BertOutput, self).__init__()
        # FeedForward层映射后的intermediate_size维度，再映射到，hidden_size维度
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # 线性映射
        hidden_states = self.dense(hidden_states)
        # dropout
        hidden_states = self.dropout(hidden_states)
        # 残差连接和layerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 8、一个transformer block
class BertLayer(nn.Module):
    """
    一个transformer block
    """

    def __init__(self, config):
        super(BertLayer, self).__init__()
        # Attention层
        self.attention = BertAttention(config)
        # FeedForward层
        self.intermediate = BertIntermediate(config)
        # FeedForward后的Add＆LayerNorm层
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, get_attention_matrices=False):
        # Attention层(包括残差连接和LayerNorm)
        attention_output, attention_matrices = self.attention(hidden_states, attention_mask,
                                                              get_attention_matrices=get_attention_matrices)
        # FeedForward层
        intermediate_output = self.intermediate(attention_output)
        # FeedForward后的Add＆LayerNorm层
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_matrices


# 9、transformer blocks * N
class BertEncoder(nn.Module):
    """
    transformer blocks * N
    """

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        # 一个transformer block
        layer = BertLayer(config)
        # 复制N个transformer block
        # num_hidden_layers个
        # copy.deepcopy()，Deep copy operation on arbitrary Python objects.
        # nn.ModuleList()，Holds submodules in a list.
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, get_attention_matrices=False):
        """

        :param hidden_states:
        :param attention_mask:
        :param output_all_encoded_layers: 是否输出每一个transformer block的隐藏层计算结果
        :param get_attention_matrices:
        :return: all_encoder_layers, all_attention_matrices
        """

        # 注意力矩阵可视化list
        all_attention_matrices = []
        # N个transformer block列表
        all_encoder_layers = []
        # 遍历N个transformer block
        for layer_module in self.layer:
            # 每一个transformer block
            hidden_states, attention_matrices = layer_module(hidden_states, attention_mask,
                                                             get_attention_matrices=get_attention_matrices)
            # 若output_all_encoded_layers为True
            # 添加每一个transformer block的隐藏层计算结果，添加两个到list
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                all_attention_matrices.append(attention_matrices)

        # 若output_all_encoded_layers为False
        # 只添加最后一个transformer block的隐藏层计算结果
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_attention_matrices.append(attention_matrices)

        # all_encoder_layers, all_attention_matrices
        return all_encoder_layers, all_attention_matrices


# 10、Pooler，取CLS（called by 15、bert模型）
class BertPooler(nn.Module):
    """
    Pooler，把隐藏层中对应#CLS#的token，的一条提取出来，
    相当于cnn的池化，降维压缩采样，
    一个Linear线形层，加一个Tanh()的激活函数，用来池化BertEncoder的输出。

    :return: pooled_output
    """

    def __init__(self, config):
        super(BertPooler, self).__init__()
        # 线性映射hidden_size到hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数，用Tanh
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # [batch_size, seq_length, embedding_dimension(即all_head_size)]
        # 第一维全取，第二维取第一channel，可能第三维全取
        # （可以理解为，首先全取第一维，然后在第二维取第一channel，最后的第三维全取）
        # We "pool" the model by simply taking the hidden state corresponding to the first token.
        first_token_tensor = hidden_states[:, 0]
        # 线性映射
        pooled_output = self.dense(first_token_tensor)
        # 激活
        pooled_output = self.activation(pooled_output)
        return pooled_output


# 11、封装：线性映射, 激活, LayerNorm（called by 12、"decoder"，用来进行MLM的预测）
class BertPredictionHeadTransform(nn.Module):
    """
    线性映射, 激活, LayerNorm

    :return: hidden_states
    """

    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        # 线性映射
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活
        self.transform_act_fn = ACT2FN[config.hidden_act]
        # LayerNorm
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        # 线性映射
        hidden_states = self.dense(hidden_states)
        # 激活
        hidden_states = self.transform_act_fn(hidden_states)
        # LayerNorm
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# 12、"decoder"，为了进行MLM的预测（called by 13、通过隐藏层，输出MLM的预测和NSP的预测）
class BertLMPredictionHead(nn.Module):
    """
    decoder，为了进行MaskedLM的预测

    :return: hidden_states
    """

    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()

        # 线性映射, 激活, LayerNorm
        self.transform = BertPredictionHeadTransform(config)

        # "decoder"
        """下面是创建一个线性映射层, 
        把transformer block输出的[batch_size, seq_len, embed_dim]
        映射为[batch_size, seq_len, vocab_size], 
        也就是把最后一个维度数映射成字典中字的数量, 
        用来进行MaskedLM的预测，获取MaskedLM的预测结果，这句很关键，
        注意这里其实也可以直接，矩阵乘embedding矩阵的转置, 
        但一般情况下我们要随机初始化新的一层参数
        """
        """
        The output weights are the same as the input embeddings, but there is
        an output-only bias for each token.
        """
        # bert_model_embedding_weights.size(1)线性映射到bert_model_embedding_weights.size(0)
        # 第二维度数线性映射到第一维度数，不加偏置
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1), bert_model_embedding_weights.size(0),
                                 bias=False)
        # decoder权重，赋值之前用的encoder权重
        self.decoder.weight = bert_model_embedding_weights
        # 偏置，赋值为第一维度数的全0矩阵
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        # 线性映射, 激活, LayerNorm
        hidden_states = self.transform(hidden_states)
        # 权重，偏置
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


# 13、通过隐藏层，输出MLM的预测和NSP的预测（called by 16、bert模型预训练）
class BertPreTrainingHeads(nn.Module):
    """
    通过隐藏层，输出MLM的预测和NSP的预测

    :return: prediction_scores, seq_relationship_score
    """

    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()

        # 用来进行MLM的预测，hidden_states经过线性映射, 激活, LayerNorm, 权重，偏置
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        # 用来进行NSP的预测，hidden_size线性映射为2分类
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        # 通过sequence_output，进行MLM的预测，获取scores
        prediction_scores = self.predictions(sequence_output)
        # 通过pooled_output，进行NSP的预测，获取scores
        seq_relationship_score = self.seq_relationship(pooled_output)

        return prediction_scores, seq_relationship_score


# 14、初始化模型参数
class BertPreTrainedModel(nn.Module):
    """
    用来初始化模型参数

    An abstract class to handle weights initialization and a simple interface
    for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()

        # 判断config是否是BertConfig实例
        if not isinstance(config, BertConfig):
            # 不是的话抛出异常
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))

        # config
        self.config = config

    def init_bert_weights_bias(self, module):
        """
        初始化权重，与偏置

        Initialize the weights.
        """

        # 判断module

        # 是nn.Linear实例
        # 加括号是因为源码用的是(nn.Linear, nn.Embedding)元组
        if isinstance(module, (nn.Linear)):
            # 初始化，线性映射层的参数，为正态分布
            # std=self.config.initializer_range，初始化模型参数的标准差
            # mean=0.0，初始化模型参数的均值
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        # 是BertLayerNorm实例
        elif isinstance(module, BertLayerNorm):
            # 初始化LayerNorm中的alpha为全1, beta为全0
            # 偏置全0
            module.bias.data.zero_()
            # 权重全1
            module.weight.data.fill_(1.0)

        # 是nn.Linear实例，且module.bias不为None
        if isinstance(module, nn.Linear) and module.bias is not None:
            # 初始化偏置为0
            module.bias.data.zero_()


# 15、bert模型
class BertModel(BertPreTrainedModel):
    """
    bert模型

    :return: 最后一个hidden_states，最后一个hidden_states的pooler

    BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        # embedding
        self.embeddings = BertEmbeddings(config)
        # transformer blocks * N
        self.encoder = BertEncoder(config)
        # Pooler
        self.pooler = BertPooler(config)
        # apply()，递归地将里面的函数，应用于每个子模块以及self，典型的用法包括初始化模型的参数
        self.apply(self.init_bert_weights_bias)

    def forward(self, input_ids, positional_enc, token_type_ids=None, attention_mask=None,
                output_all_encoded_layers=True, get_attention_matrices=False):

        # 若attention_mask为none
        if attention_mask is None:
            # (input_ids > 0)，返回相同维度特性的布尔值
            # input_ids [batch_size, sequence_length]
            # attention_mask [batch_size, sequence_length]
            attention_mask = (input_ids > 0)

        # 若token_type_ids为none
        if token_type_ids is None:
            # input_ids相同维度，初始化全0
            token_type_ids = torch.zeros_like(input_ids)

        # 注意力矩阵mask: [batch_size, 1, 1, seq_length]
        # unsqueeze()，指定位置插入attention_mask
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # 给注意力矩阵里padding的无效区域加一个很大的负数的偏置, 为了使softmax之后这些无效区域仍然为0, 不参与后续计算，相应的score值变为极小
        # self.parameters()，Returns an iterator over module parameters.
        # next()，Return the next item from the iterator.
        # torch.dtype是表示torch.Tensor的数据类型的对象
        # to(dtype=)，应该是转换到某数据类型，原来是布尔值
        # 若是1.0，数值基本不变；若是0.0，变成一个很大负数的的偏置
        # Since attention_mask is 1.0 for positions
        # we want to attend and 0.0 for masked positions,
        # this operation will create a tensor
        # which is 0.0 for positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax,
        # this is effectively the same as removing these entirely.
        # fp16 compatibility
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding层
        embedding_output = self.embeddings(input_ids, positional_enc, token_type_ids)
        # 经过所有定义的transformer block之后的输出
        encoded_layers, all_attention_matrices = self.encoder(embedding_output,
                                                              extended_attention_mask,
                                                              output_all_encoded_layers=output_all_encoded_layers,
                                                              get_attention_matrices=get_attention_matrices)
        # 是否输出所有层的注意力矩阵用于可视化，返回
        if get_attention_matrices:
            return all_attention_matrices

        # 获取pooled_output
        # [-1]为最后一个transformer block的隐藏层的计算结果hidden_states
        sequence_output = encoded_layers[-1]
        # pooled_output为隐藏层中#CLS#对应的token的一条向量
        pooled_output = self.pooler(sequence_output)

        # 获取encoded_layers=最后一个transformer block的隐藏层的计算结果hidden_states
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # 最后一个hidden_states，最后一个hidden_states的pooler
        return encoded_layers, pooled_output


# 16、bert模型预训练（15、bert模型；13、通过隐藏层，输出MLM的预测和NSP的预测）
class BertForPreTraining(BertPreTrainedModel):
    """
    bert模型预训练

    :return: mlm_preds, next_sen_preds

    BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)

        # config
        self.bert = BertModel(config)
        # bert的预训练，传入字嵌入初始化权重，获取cls那一条向量
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        # 设置初始化bert模型参数，应用到每一个子模块
        self.apply(self.init_bert_weights_bias)
        # 字典字数
        self.vocab_size = config.vocab_size
        # NSP用交叉熵损失函数
        self.next_loss_func = CrossEntropyLoss()
        # MLM用交叉熵损失函数
        # ignore_index(int)，忽略某一类别，不计算其loss，其loss会为0，因为计算也为0
        self.mlm_loss_func = CrossEntropyLoss(ignore_index=0)

    def compute_loss(self, predictions, labels, num_class=2, ignore_index=-100):
        """
        关于labels：
            如果labels不是None（训练时）：输出的是分类的交叉熵
            如果labels是None（评价时）：输出的是shape为[batch_size, num_labels]估计值
        """
        # 计算损失
        loss_func = CrossEntropyLoss(ignore_index=ignore_index)
        # 计算预测（自动合成维度，二分类）和正确（自动合成维度）之前的交叉熵
        # view(-1)
        # 把原先tensor中的数据按照行优先的顺序排成一个一维的数据（这里应该是因为要求地址是连续存储的），然后按照参数组合成其他维度的tensor
        # -1就代表这个位置由其他位置的数字来推断，只要在不致歧义的情况的下，view参数就可以推断出来
        return loss_func(predictions.view(-1, num_class), labels.view(-1))

    def forward(self, input_ids, positional_enc, token_type_ids=None, attention_mask=None,
                masked_lm_labels=None, next_sentence_label=None):
        # 走bert模型
        sequence_output, pooled_output = self.bert(input_ids, positional_enc, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        # 通过隐藏层，输出MLM的预测和NSP的预测
        mlm_preds, next_sen_preds = self.cls(sequence_output, pooled_output)

        # 返回mlm_preds, next_sen_preds
        return mlm_preds, next_sen_preds
