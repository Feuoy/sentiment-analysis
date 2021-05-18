import torch
from abc import abstractmethod
from numpy import inf
from logger.visualization import TensorboardWriter


class BaseTrainer:
    """
    训练器基类

    Base class for all trainers
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        """

        :param model: 模型
        :param criterion: 损失标准
        :param metric_ftns: 度量工具函数（评价指标）
        :param optimizer: 优化器
        :param config: 配置
        """

        # 配置
        self.config = config
        # logger
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # 准备计算代理，返回self.device和gpu列表
        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        # 模型丢进计算代理
        self.model = model.to(self.device)

        # # 1、单卡，把下面注释掉；2、多卡并行，不注释下面，
        # # 如果gpu数大于1
        # if len(device_ids) > 1:
        #     # 实现了并行计算
        #     # DataParallel()，Implements data parallelism at the module level.
        #     self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        # 损失标准
        self.criterion = criterion
        # 度量工具函数
        """
        这个
        "metrics": [
            "binary_accuracy",
            "binary_f1",
            "binary_auc",
        ],
        """
        self.metric_ftns = metric_ftns
        # 优化器
        self.optimizer = optimizer

        # 训练器配置
        cfg_trainer = config['trainer']
        # 当前轮开始的epoch
        self.start_epoch = 1
        # 这轮一共要训练的epoch
        self.epochs = cfg_trainer['epochs']
        # 保存周期
        self.save_period = cfg_trainer['save_period']

        # 监视曲线
        """
        "monitor": "min val_loss"
        json.get('monitor', 'off')，如果没'monitor'键，会返回'off'        
        """
        self.monitor = cfg_trainer.get('monitor', 'off')

        # 配置监视曲线，来保存最好模型
        # configuration to monitor model performance and save best

        # 若监视曲线，是关闭的
        if self.monitor == 'off':
            # 倾向模式，关闭
            self.mnt_mode = 'off'
            # 当前最好倾向，0
            self.mnt_best = 0

        # 若监视曲线，是开启的
        else:
            # 倾向模式"min"，
            # 度量曲线"val_loss"，
            # "monitor": "min val_loss"
            self.mnt_mode, self.mnt_metric = self.monitor.split()

            # 确保倾向模式在['min', 'max']
            assert self.mnt_mode in ['min', 'max']

            # 当前最好倾向，初始化
            """
            倾向min，赋inf无穷大，
            反之则反

            站的越高，看的越远
            反之则反：站的越低，看的越近（句子对反）
            反之亦然：看的越远，高的越高（前后句反）            
            """
            self.mnt_best = inf if self.mnt_mode == 'min' else -inf

            # 提前停止
            """
            是几个模型没有提高之后的停止计数值
            "early_stop": 10,
            如果'early_stop'没有值，那么inf
            """
            self.early_stop = cfg_trainer.get('early_stop', inf)

        # 可视化Writer实例
        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        # 检查点模型保存路径
        self.checkpoint_dir = config.save_dir
        # print(self.checkpoint_dir)

        # 如果设置了重启路径
        if config.resume is not None:
            # 起用重启
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        训练一个epoch，抽象方法

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        训练

        Full training logic
        """

        # 没有再提高的epoch计数器
        not_improved_count = 0

        # 遍历（start_epoch）到（epochs + 1）
        for epoch in range(self.start_epoch, self.epochs + 1):

            # 单个训练
            result = self._train_epoch(epoch)

            # 记录
            # save logged informations into log dict
            log = {'[epoch]': epoch}
            log.update(result)

            # 打印记录到屏幕
            # print logged informations to the screen
            for key, value in log.items():
                # （'binary_accuracy': 值）
                self.logger.info('    {:25s}: {}'.format(str(key), value))

            """
            通过度量，评价模型性能，来保存最佳模型
            evaluate model performance according to configured metric, save best checkpoint as model_best
            """

            # 是否当前最佳标记
            best = False

            # 如果监视曲线开了
            if self.mnt_mode != 'off':
                try:
                    # 判断这个epoch有没有提高，布尔
                    """
                    倾向min：监视曲线度量，<=，当前最好倾向（没有再提高了）
                    反则反之
                    check whether model performance improved or not, according to specified metric(mnt_metric)                    
                    """
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    # 监视度量工具没有找到，模型性能监视不可用
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    # 倾向，关闭
                    self.mnt_mode = 'off'
                    # improved，未提高
                    improved = False

                # 如果还在提高
                if improved:
                    # 更新当前最好倾向，为，监视当前度量
                    self.mnt_best = log[self.mnt_metric]
                    # 不再提高计数，清零（防止有的epoch确实会偶尔降低准确率）
                    not_improved_count = 0
                    # 当前最佳标记，true
                    best = True

                # 不再提高计数+1
                else:
                    not_improved_count += 1

                # 如果不再提高计数，大于，提前停止计数
                if not_improved_count > self.early_stop:
                    # log：有效性能不再提高，已经有self.early_stop个epoch了，训练停止
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    # 训练停止
                    break

            # epoch序数，到了，保存周期（"save_period": 1）
            if epoch % self.save_period == 0:
                # 保存模型
                # save_best=best，的best，最小损失的那个模型
                self._save_checkpoint(epoch, save_best=best)

    def _prepare_device(self, n_gpu_use):
        """
        准备计算代理
        setup GPU device if available, move model into configured device
        """

        # gpu个数
        n_gpu = torch.cuda.device_count()

        # 设置使用gpu数大于0，但没有gpu
        if n_gpu_use > 0 and n_gpu == 0:
            # 提示
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            # n_gpu_use，赋0
            n_gpu_use = 0

        # n_gpu_use大于拥有gpu个数
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            # n_gpu_use，用最大gpu个数
            n_gpu_use = n_gpu

        # 如果n_gpu_use大于0，用第几号gpu
        device = torch.device('cuda:{}'.format(self.config.config['device_id']) if n_gpu_use > 0 else 'cpu')

        # gpu列表
        """
        map(func, *iterables) --> map object

        Make an iterator that computes the function using arguments from
        each of the iterables.  Stops when the shortest iterable is exhausted.
        """
        list_ids = list(map(lambda x: int(x), self.config.config['device_id'].split(',')))

        # 返回device和gpu列表
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        保存checkpoints

        :param epoch: 当前epoch序数
        current epoch number
        :param log: 记录当前epoch信息
        logging information of the epoch
        :param save_best: 是否作为最佳model
        if True, rename the saved checkpoint to 'model_best.pth'
        """

        # 模型的"model_arch"
        """
        "model_arch": {
            "type": "RNN",
            "args": {
                "rnn_type": "lstm",
                "embedding_dim": 300,
                "hidden_dim": 256,
                "output_dim": 1,
                "n_layers":2,
                "bidirectional": true,
                "dropout": 0,
                "batch_first": false
            }
        },
        """
        arch = type(self.model).__name__

        # 保存状态
        state = {
            'arch': arch,  # 类型与架构（"model_arch"）
            'epoch': epoch,  # epoch
            'state_dict': self.model.state_dict(),  # 状态
            'optimizer': self.optimizer.state_dict(),  # 优化器
            'monitor_best': self.mnt_best,  # 当前最好倾向
            'config': self.config  # 配置文件
        }
        # 文件路径
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        # 保存
        torch.save(state, filename)

        # 记日志
        self.logger.info("Saving checkpoint: {} ...".format(filename))

        # 如果是，保存为最佳模型
        if save_best:
            # 最好文件名
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            # 再保存一个
            torch.save(state, best_path)
            # 记日志
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        由保存的checkpoints重新开始，要传入参数产能使用
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """

        # 检查点模型的路径
        resume_path = str(resume_path)
        # log：加载检查点模型
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        # 加载
        checkpoint = torch.load(resume_path)
        # start_epoch赋为，上次的epoch序数递增
        self.start_epoch = checkpoint['epoch'] + 1
        # 当前最好倾向，赋值
        self.mnt_best = checkpoint['monitor_best']

        """
        从checkpoint加载架构，到现在的self.model
        load architecture params from checkpoint.
        """
        # 若checkpoint和config，的'arch'不相同
        if checkpoint['config']['arch'] != self.config['arch']:
            # log：可能会出现异常
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        # 加载checkpoint
        self.model.load_state_dict(checkpoint['state_dict'])

        """
        加载优化器
        load optimizer state from checkpoint only when optimizer type is not changed.        
        """
        # 若checkpoint的优化器类型，不同于config的
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            # log：优化器可能不会工作
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            # 加载优化器
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        """
        log：Checkpoint加载成功，从epochNum开始重启训练
        """
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
