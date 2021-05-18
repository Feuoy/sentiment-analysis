import numpy as np
import torch
from base.base_trainer import BaseTrainer
from utils.util import inf_loop, MetricTracker
from time import time


class Trainer(BaseTrainer):
    """
    训练器，继承BaseTrainer

    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, train_iter, valid_iter, test_iter=None,
                 lr_scheduler=None, len_epoch=None):
        """

        :param model:
        :param criterion:
        :param metric_ftns:
        :param optimizer:
        :param config:
        :param train_iter: 训练集迭代器
        :param valid_iter: 验证集迭代器
        :param test_iter: 测试集迭代器，没有用测试集，test_iter=None,
        :param lr_scheduler: 学习率表
        :param len_epoch: epoch长度（一次使用训练集的长度）
        """

        # 调用父类初始化先
        super().__init__(model, criterion, metric_ftns, optimizer, config)

        self.config = config
        self.train_iter, self.valid_iter, self.test_iter = train_iter, valid_iter, test_iter

        # 如果len_epoch为none，那么就用训练集长度
        # epoch-based training
        if len_epoch is None:
            self.len_epoch = len(self.train_iter)

        # 不为none，那么用传入的len_epoch
        # iteration-based training
        else:
            # 放在self.data_loader，用来控制要训练集重复的长度，为无限循环的train_iter
            self.data_loader = inf_loop(train_iter)
            # len_epoch，用传入的
            self.len_epoch = len_epoch

        # 是否做验证，如果验证集不为空
        self.do_validation = self.valid_iter is not None
        # 学习率表
        self.lr_scheduler = lr_scheduler
        # log记录步长，训练集的batch_size的平方根取整
        self.log_step = int(np.sqrt(train_iter.batch_size))

        # 训练集、验证集的，度量记录对象
        # 传入'loss'和self.metric_ftns
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        训练一个epoch
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        # 开始时间
        t1 = time()
        # 模型训练
        self.model.train()
        # 清空训练器度量记录对象值
        self.train_metrics.reset()

        # 遍历self.train_iter（一个子次，为一个batch）
        for batch_idx, batch_data in enumerate(self.train_iter):

            # 一些：文本，bert的masks（None），文本长度，分类标签
            input_token_ids, bert_masks, seq_lens, class_label = batch_data

            # 优化器梯度清零
            self.optimizer.zero_grad()

            # 模型输出
            """
            squeeze(1)，
            将输入张量形状中的1去除并返回，
            如果输入是形如(A×1×B×1×C×1×D)，那么输出形状就为： (A×B×C×D)
            """
            output = self.model(input_token_ids, bert_masks, seq_lens).squeeze(1)

            # 计算损失
            loss = self.criterion[0](output, class_label)

            # 损失回传，反向传播
            loss.backward()
            # 优化器更新参数
            """
            step()，
            Performs a single optimization step.
            """
            self.optimizer.step()

            # 可视化当前训练进度：上一epoch乘以len_epoch，加当前batch_idx
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            # 更新损失记录（这个batch内）
            self.train_metrics.update('loss', loss.item())

            # 更新其它度量记录（这个batch内）
            for met in self.metric_ftns:
                # 每个度量函数更新度量
                self.train_metrics.update(met.__name__, met(output, class_label))

            # 每个记录步长，记录一次
            if batch_idx % self.log_step == 0:
                # log：【epoch，当前epoch-batch进度条，loss】
                self.logger.debug(
                    'Train Epoch: {} {} Loss: {:.6f}'.format(epoch, self._progress(batch_idx), loss.item()))

            # 如果batch_idx到了规定的self.len_epoch，也要停掉
            if batch_idx == self.len_epoch:
                break

        """
        # ##############################################################
        # 再说，横坐标跟batchsize、时间有关
        # bert的时候
        for batch_idx, batch_data in enumerate(self.train_iter):
            input_token_ids, bert_masks, seq_lens, class_label = batch_data
            self.optimizer.zero_grad()
            output = self.model(input_token_ids, bert_masks, seq_lens).squeeze(1)
            if (batch_idx + 1) % 8 == 1:
                output_one = output.clone()
                class_labels_one = class_label.clone()
            elif (batch_idx + 1) % 8 == 2:
                output_two = output.clone()
                class_labels_two = class_label.clone()
            elif (batch_idx + 1) % 8 == 3:
                output_three = output.clone()
                class_labels_three = class_label.clone()
            elif (batch_idx + 1) % 8 == 4:
                output_four = output.clone()
                class_labels_four = class_label.clone()
            elif (batch_idx + 1) % 8 == 5:
                output_five = output.clone()
                class_labels_five = class_label.clone()
            elif (batch_idx + 1) % 8 == 6:
                output_six = output.clone()
                class_labels_six = class_label.clone()
            elif (batch_idx + 1) % 8 == 7:
                output_seven = output.clone()
                class_labels_seven = class_label.clone()
            else:
                pred_tensor = torch.cat((output_one, output_two, output_three, output_four,
                                         output_five, output_six, output_seven, output), 0)
                label_tensor = torch.cat((class_labels_one, class_labels_two, class_labels_three, class_labels_four,
                                          class_labels_five, class_labels_six, class_labels_seven, class_label), 0)
                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(pred_tensor, label_tensor))

                loss = self.criterion[0](pred_tensor, label_tensor)
                loss.backward()
                self.optimizer.step()
                self.train_metrics.update('loss', loss.item())
                self.writer.set_step((epoch - 1) * self.len_epoch + int((batch_idx + 1)/8))

            loss_this_batch = self.criterion[0](output, class_label)
            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    'Train Epoch: {} {} Loss: {:.6f}'.format(epoch, self._progress(batch_idx), loss_this_batch.item()))
                    # 'Train Epoch: {} {} Loss: {:.6f}'.format(epoch, self._progress(batch_idx), loss.item()))
            if batch_idx == self.len_epoch:
                break
        # ##############################################################
        """

        # 记录，获取度量结果（这个epoch的）

        log = self.train_metrics.result()

        # 若做验证集
        if self.do_validation:
            # 该epoch做验证
            val_log = self._valid_epoch(epoch)
            # 更新log（添加）
            # 'val_' + k: v
            log.update(**{
                'val_' + k: v
                for k, v in val_log.items()
            })

        # 若学习率表还有
        if self.lr_scheduler is not None:
            # 走下一学习率
            self.lr_scheduler.step()

        # 打印训练这个epoch花费的时间，
        print('spending time of this epoch :', time() - t1)

        # 返回log（average loss and metric in this epoch.）
        return log

    def _valid_epoch(self, epoch):
        """
        验证一个epoch
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """

        # 模型验证
        """
        Sets the module in evaluation mode，
        不启用 BatchNorm 和 Dropout


        BatchNormalization
        不要做归一化满足独立同分布，
        对网络中间的每层进行归一化处理，并且使用变换重构（Batch Normalization Transform）保证每层提取的特征分布不会被破坏

        关于torch.no_grad()和model.eval()，
        Use both. They do different things, and have different scopes.
        with torch.no_grad()： disables tracking of gradients in autograd.
        model.eval()： changes the forward() behaviour of the module it is called upon. eg, it disables dropout 
        and has batch norm use the entire population statistics
        """
        self.model.eval()
        # 验证集度量实例值清空
        self.valid_metrics.reset()

        # 停掉计算自动梯度，下面的操作
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.valid_iter):
                input_token_ids, bert_masks, seq_lens, class_label = batch_data
                output = self.model(input_token_ids, bert_masks, seq_lens).squeeze(1)
                loss = self.criterion[0](output, class_label)
                self.writer.set_step((epoch - 1) * len(self.valid_iter) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, class_label))

            """
            # ##################################################################################################
            # bert的时候
            for batch_idx, batch_data in enumerate(self.valid_iter):
                input_token_ids, bert_masks, seq_lens, class_label = batch_data
                output = self.model(input_token_ids, bert_masks, seq_lens).squeeze(1)
                if (batch_idx + 1) % 8 == 1:
                    output_one = output.clone()
                    class_labels_one = class_label.clone()
                elif (batch_idx + 1) % 8 == 2:
                    output_two = output.clone()
                    class_labels_two = class_label.clone()
                elif (batch_idx + 1) % 8 == 3:
                    output_three = output.clone()
                    class_labels_three = class_label.clone()
                elif (batch_idx + 1) % 8 == 4:
                    output_four = output.clone()
                    class_labels_four = class_label.clone()
                elif (batch_idx + 1) % 8 == 5:
                    output_five = output.clone()
                    class_labels_five = class_label.clone()
                elif (batch_idx + 1) % 8 == 6:
                    output_six = output.clone()
                    class_labels_six = class_label.clone()
                elif (batch_idx + 1) % 8 == 7:
                    output_seven = output.clone()
                    class_labels_seven = class_label.clone()
                else:
                    pred_tensor = torch.cat((output_one, output_two, output_three, output_four,
                                             output_five, output_six, output_seven, output), 0)
                    label_tensor = torch.cat((class_labels_one, class_labels_two, class_labels_three, class_labels_four,
                                              class_labels_five, class_labels_six, class_labels_seven, class_label), 0)
                    for met in self.metric_ftns:
                        self.valid_metrics.update(met.__name__, met(pred_tensor, label_tensor))
                    loss = self.criterion[0](output, class_label)
                    self.valid_metrics.update('loss', loss.item())
                    self.writer.set_step((epoch - 1) * len(self.valid_iter) + int((batch_idx + 1)/8), 'valid')
                # ##################################################################################################
            """

        # 遍历self.model的已命名参数（？），每个都添加直方图，到self.writer
        """
        add histogram of model parameters to the tensorboard
        named_parameters()，
        Returns an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.
        """
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        # 【log=获取度量结果（这个epoch的）】
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        """
        一个epoch，训练过程中，的进度条

        :param batch_idx:
        :return:
        """

        # 如果self.train_iter有'n_samples'属性
        # （iteration-based training）
        if hasattr(self.train_iter, 'n_samples'):
            # batch_idx乘以batch_size
            current = batch_idx * self.data_loader.batch_size
            # n个样本量
            total = self.data_loader.n_samples

        # 否则
        # （epoch-based training）
        else:
            # batch索引
            current = batch_idx
            # epoch长度
            total = self.len_epoch

        # 当前进度数，所有进度数，进度百分比
        return '[{}/{} ({:.0f}%)]'.format(current, total, 100.0 * current / total)
