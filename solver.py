#! encoding=utf8
# pylint: skip-file
import numpy as np
import mxnet as mx
import time
import logging
from collections import namedtuple
from mxnet import optimizer as opt
from mxnet.optimizer import get_updater
from mxnet import metric
from metric_mIOU import MeanIoU

# Parameter to pass to batch_end_callback
BatchEndParam = namedtuple('BatchEndParams', ['epoch', 'nbatch', 'eval_metric'])


class Solver(object):
    def __init__(self, symbol, ctx=None,
                 begin_epoch=0, num_epoch=None,
                 arg_params=None, aux_params=None,
                 optimizer='sgd', **kwargs):
        self.symbol = symbol
        if ctx is None:
            ctx = mx.cpu()
        self.ctx = ctx
        self.begin_epoch = begin_epoch
        self.num_epoch = num_epoch

        self.arg_params = arg_params
        self.aux_params = aux_params
        self.grad_params = None

        self.optimizer = optimizer
        self.updater = None
        self.executor = None
        self.kwargs = kwargs.copy()

    def fit(self, train_data,
            eval_data=None,
            eval_metric='acc',
            period=['train', 'val'],
            to_eval_train=True,
            grad_req='write',
            epoch_end_callback=None,
            batch_end_callback=None,
            kvstore='local',
            logger=None):

        if logger is None:
            logger = logging
        logging.info('Start training with %s', str(self.ctx))
        # region 1. 准备参数，包括输入数据和标签数据
        # FCN的参数名
        arg_names = self.symbol.list_arguments()
        # FCN的参数形状
        # print train_data.provide_data[0]
        arg_shapes, out_shapes, aux_shapes = self.symbol.infer_shape(data=train_data.provide_data[0][1])
        # arg_shapes, out_shapes, aux_shapes = self.symbol.infer_shape(data=(1, 3,
        #                                                                    train_data.resize_size[0],
        #                                                                    train_data.resize_size[1],
        #                                                                    ))
        # print train_data.provide_data[0][1]
        # quit()
        # 输入数据和标签数据
        data_name = train_data.provide_data[0][0]
        label_name = train_data.provide_label[0][0]
        # print data_name, label_name
        # input_names = [data_name, label_name]
        # batch_size, channel, h, w
        # data_shape = train_data.provide_data[0][1]
        self.arg_params[data_name] = mx.nd.empty(
            train_data.provide_data[0][1],
            self.ctx
        )
        # # batch_size, h*w
        self.arg_params[label_name] = mx.nd.empty(
            train_data.provide_label[0][1],
            self.ctx)
        # quit()
        # 其他参数
        aux_names = self.symbol.list_auxiliary_states()
        self.aux_params = {k: mx.nd.zeros(s) for k, s in zip(aux_names, aux_shapes)}
        # endregion

        # region 2.准备参数的梯度
        if grad_req != 'null':
            self.grad_params = {}
            for name, shape in zip(arg_names, arg_shapes):
                if not (name.endswith('data') or name.endswith('label')):
                    # print name,shape
                    self.grad_params[name] = mx.nd.zeros(shape, self.ctx)
        else:
            self.grad_params = None
        # endregion
        # print self.arg_params
        # region 3. 绑定模型参数 和 模型的输出
        self.executor = self.symbol.bind(
            self.ctx,
            self.arg_params,
            args_grad=self.grad_params,
            grad_req=grad_req,
            aux_states=self.aux_params
        )
        # quit()
        assert len(self.symbol.list_arguments()) == len(self.executor.grad_arrays)
        # 绑定输出变量
        output_dict = {}
        output_buff = {}
        for key, arr in zip(self.symbol.list_outputs(), self.executor.outputs):
            # print key, arr
            output_dict[key] = arr
            output_buff[key] = mx.nd.empty(arr.shape, ctx=mx.cpu())
        # endregion

        # region 4. 设置优化器
        self.optimizer = opt.create(
            self.optimizer,
            rescale_grad=1.0 / train_data.batch_size,
            **self.kwargs
        )
        self.updater = get_updater(self.optimizer)
        # 需要更新梯度的参数
        update_dict = {name: nd for name, nd in zip(self.symbol.list_arguments(),
                                                    self.executor.grad_arrays) if nd is not None}
        # endregion

        # region 5. 设置评价尺度
        if eval_metric == 'acc':
            eval_metric = metric.create(eval_metric)
        elif eval_metric == 'meanIOU':
            eval_metric = MeanIoU(c=1,)
        # endregion

        for epoch in range(self.begin_epoch, self.num_epoch):
            # region begin training
            if 'train' in period:
                logger.info(" in train process...")
                all_start = time.time()
                nbatch = 0
                train_data.reset()
                eval_metric.reset()
                for data in train_data:
                    nbatch += 1
                    # all_start = time.time()
                    # region 1. 准备 batch 数据
                    # start = time.time()
                    self.arg_params[data_name][:] = data.data[0]
                    # end = time.time()
                    # print end-start
                    # label_shape = data.label[0].shape
                    # print label_shape
                    self.arg_params[label_name][:] = data.label[0]
                    # end = time.time()
                    # print 'prepare data and label time: %s s' % (end - start)
                    # quit()
                    # print self.arg_params[label_name][:]
                    # endregion

                    # region 2. forward
                    # start = time.time()
                    self.executor.forward(is_train=True)
                    # end = time.time()
                    # print 'forward time: %s s' % (end - start)

                    # endregion

                    # region 3. backward
                    # start = time.time()
                    self.executor.backward()
                    for key, arr in update_dict.items():
                        if key != "bigscore_weight":
                            # 参数名,梯度, 权重
                            self.updater(key, arr, self.arg_params[key])
                            # self.executor.outputs[0].wait_to_read()
                    # end = time.time()
                    # print 'backward time: %f s' % (end - start)
                    # endregion

                    # region 4. 测评
                    # start = time.time()
                    if to_eval_train:
                        # start = time.time()
                        # 取得输出
                        for key in output_dict:
                            # print key
                            output_dict[key].copyto(output_buff[key])
                            # output_dict[key].wait_to_read()
                        # end = time.time()
                        # print 'output1 copy time: %s s' % (end - start)
                        # start = time.time()
                        pred_shape = output_buff['softmax_output'].shape
                        # print pred_shape, label_shape
                        # label = self.arg_params[label_name]
                        pred = output_buff['softmax_output'].reshape((pred_shape[0],
                                                                      pred_shape[1],
                                                                      pred_shape[2]*pred_shape[3]
                                                                      ))
                        # pred = pred.copyto(self.ctx)
                        # print pred.shape
                        label = data.label[0]
                        # quit()
                        # end = time.time()
                        # print 'output copy2 time: %s s' % (end - start)
                        # 更新评价
                        eval_metric.update([label], [pred])
                    batch_end_params = BatchEndParam(epoch=epoch,
                                                     nbatch=nbatch,
                                                     eval_metric=eval_metric if to_eval_train else None,
                                                     )
                    batch_end_callback(batch_end_params)
                    # end = time.time()
                    # print '测评 time: %s s' % (end - start)
                    # endregion
                    # all_end = time.time()
                    # print 'all time: %s s' % (all_end - all_start)
                    # if nbatch > 1:
                    #     quit()
                if epoch_end_callback is not None:
                    epoch_end_callback(epoch, self.symbol, self.arg_params, self.aux_params)

                # all_end = time.time()
                # print 'all time1: %s s' % (all_end - all_start)
                if to_eval_train:
                    name, value = eval_metric.get()
                    logger.info("                     --->Epoch[%d] Train-%s=%f", epoch, name, value)
                logger.info('train time per epoch: %f s' % (time.time() - all_start))
            # endregion
            # evaluation
            if 'val' in period and eval_data:
                logger.info(" in eval process...")
                nbatch = 0
                eval_data.reset()
                eval_metric.reset()
                # all_start = time.time()
                for data in eval_data:
                    nbatch += 1
                    # label_shape = data.label.shape

                    self.arg_params[data_name][:] = data.data[0]
                    self.arg_params[label_name][:] = data.label[0]

                    self.executor.forward(is_train=False)
                    pred_shape = self.executor.outputs[0].shape

                    cpu_output_array = mx.nd.empty(pred_shape)
                    self.executor.outputs[0].copyto(cpu_output_array)

                    label = data.label[0]

                    pred = cpu_output_array.reshape((pred_shape[0],
                                                     pred_shape[1],
                                                     pred_shape[2] * pred_shape[3]))

                    eval_metric.update([label], [pred])

                    batch_end_params = BatchEndParam(epoch=epoch,
                                                     nbatch=nbatch,
                                                     eval_metric=None,
                                                     )
                    batch_end_callback(batch_end_params)

                    # if nbatch>200:
                    #     quit()
                    # quit()
                    # self.executor.outputs[0].wait_to_read()
                # all_end = time.time()
                # print 'all time1: %s s' % (all_end - all_start)
                # all_start = time.time()
                name, value = eval_metric.get()
                logger.info('Epoch[%d] Validation-%s=%f', epoch, name, value)
                # all_end = time.time()
                # print 'all time2: %s s' % (all_end - all_start)
                # quit()
