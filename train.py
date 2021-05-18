import argparse
import collections
import torch
import numpy as np
from data_process import data_process as module_data_process
from torch.utils.data import dataloader as module_data_loader
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from utils.parse_config import ConfigParser
from trainer.trainer import Trainer
import transformers as optimization

"""
设置随机种子，为了可复现
fix random seeds for reproducibility
在神经网络中，参数默认是进行随机初始化的，
不同的初始化参数往往会导致不同的结果，当得到比较好的结果时我们通常希望这个结果是可以复现的，
在pytorch中，通过设置随机数种子也可以达到这么目的。

seed (int): The desired seed.
manual_seed()，Sets the seed for generating random numbers. Returns a `torch.Generator` object.

cuDNN（CUDA Deep Neural Network library），针对深度神经网络的GPU加速库
在 PyTorch 中对模型里的卷积层进行预先的优化，也就是在每一个卷积层中测试 cuDNN 提供的所有卷积实现算法，然后选择最快的那个，
这样在模型启动的时候，只要额外多花一点点预处理时间，就可以较大幅度地减少训练时间，
如果我们的网络模型一直变的话，那肯定是不能设置 cudnn.benchmark=True 的，
因为网络结构经常变，每次 PyTorch 都会自动来根据新的卷积场景做优化，
这次花费了半天选出最合适的算法出来，结果下次你结构又变了，之前就白做优化了。

torch.backends.cudnn.deterministic = True，cuDNN使用确定性卷积，
torch.backends.cudnn.benchmark = False，可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题

pytorch和numpy有接口的内在联系，pytorch内部也会调用numpy的一些函数，
所以numpy也要设置，np.random.seed(SEED)，
"""

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config, use_transformers):
    """
    训练的时候,同一种类型的模型，用同一种类型的嵌入pkl
    跑不了bert，把下载模型里面的改为config.json
    """
    # 获取这个'train'日志名，的logger
    logger = config.get_logger('train')

    # 计算代理
    device = torch.device('cuda:{}'.format(config.config['device_id']) if config.config['n_gpu'] > 0 else 'cpu')

    # data_set实例
    """
        base_dataset——data_split
        data_process——test_size
        trainer——test_iter
        train——valid_dataloader
    """
    dataset = config.init_obj('dataset', module_data_process, device=device)

    # 模型架构，然后打印到控制台
    model = config.init_obj('model_arch', module_arch, word_embedding=dataset.word_embedding)
    logger.info(model)

    # 损失标准（loss函数），准确率度量（acc函数）
    criterion = [getattr(module_loss, crit) for crit in config['loss']]
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # bert训练
    if use_transformers:
        #
        # 筛选参数
        if 'bert' in config.config['model_arch']['type'].lower():
            # transformers参数
            transformers_params = [*filter(lambda p: p.requires_grad, model.bert.parameters())]
            # 其它参数
            other_params = [*filter(lambda p: p.requires_grad,
                                    [param for name, param in model.named_parameters() if 'bert' not in name])]

        # dataloader实例
        train_dataloader = config.init_obj('data_loader', module_data_loader, dataset=dataset.train_set,
                                           collate_fn=dataset.bert_collate_fn)
        valid_dataloader = config.init_obj('data_loader', module_data_loader, dataset=dataset.test_set,
                                           collate_fn=dataset.bert_collate_fn)

        # 优化器
        optimizer = config.init_obj('optimizer',
                                    optimization,
                                    [{"params": transformers_params, 'lr': 5e-5, "weight_decay": 0},
                                     {"params": other_params, 'lr': 1e-3, "weight_decay": 0}, ]
                                    )
        # 学习率
        lr_scheduler = config.init_obj('lr_scheduler',
                                       optimization.optimization,
                                       optimizer,
                                       num_training_steps=int(
                                           len(train_dataloader.dataset) / train_dataloader.batch_size)
                                       )

    # 普通训练
    else:
        # dataloader实例
        """
        dataset.test_set用做valid_dataloader
        """
        train_dataloader = config.init_obj('data_loader', module_data_loader, dataset=dataset.train_set,
                                           collate_fn=dataset.collate_fn)
        valid_dataloader = config.init_obj('data_loader', module_data_loader, dataset=dataset.test_set,
                                           collate_fn=dataset.collate_fn)

        # 优化器
        """
        筛选model.parameters()为可训练参数（可有梯度），放进优化器
        """
        trainable_params = [*filter(lambda p: p.requires_grad, model.parameters())]
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

        # 学习率调整器
        """
        删除每一个不可训练的的学习调整器
        delete every lines containing lr_scheduler for disabling scheduler
        """
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # 一个trainer
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      train_iter=train_dataloader,
                      valid_iter=valid_dataloader,
                      lr_scheduler=lr_scheduler)

    # 训练
    trainer.train()


def run(config_file):
    # 命令行解析器
    args = argparse.ArgumentParser(description='text classification')

    # 添加命令：配置、重启、计算
    args.add_argument('-c', '--config', default=config_file, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='0,1', type=str, help='indices of GPUs to enable (default: all)')

    # 客户参数：可修改的配置
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    # 添加客户参数：学习率、批量长度
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_process;args;batch_size')
    ]
    # 配置解析器
    config = ConfigParser.from_args(args, options)
    # 打印训练的模型类型
    print(config.config['model_arch']['type'].lower())

    # 训练
    if 'bert' in config.config['model_arch']['type'].lower():
        main(config, use_transformers=True)
    else:
        main(config, use_transformers=False)


if __name__ == '__main__':
    run('configs/mlp/mlp_1h_config.json')
    # run('configs/mlp/mlp_2h_config.json')
    # run('configs/mlp/mlp_4h_config.json')
    # # ------------
    # run('configs/textcnn/textcnn_1d_config.json')
    # run('configs/textcnn/textcnn_2d_config.json')
    # # ------------
    # run('configs/rnn/rnn_config.json')
    # run('configs/rnn/lstm_config.json')
    # run('configs/rnn/gru_config.json')
    # # ------------
    # run('configs/attention/rnn_attention_config.json')
    # # ------------
    # run('configs/textrcnn/textrcnn_config.json')
    # # ------------
    # run('configs/bert/bert_config.json')
    # run('configs/bert/bert_cnn_config.json')
    # run('configs/bert/bert_rnn_attention_config.json')
    # run('configs/bert/bert_rcnn_config.json')
    # # ------------
    # run('configs/new/final_config.json')
