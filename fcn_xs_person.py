# encoding=utf8
# pylint: skip-file

import sys, os
import argparse
import mxnet as mx
import numpy as np
import logging
import symbol_fcnxs_person as symbol_fcnxs
import symbol_fcnxs_atrous_person
import init_fcnxs
from data_person import get_train_val_iter
from solver import Solver
# region 在这里设置参数
speedometer_freq = 100
period = [
    'train',
    'val',
]
resize_size = (256, 256)
buffer_image_set = True
use_record_data = True
data_type = 'coco_portrait_pascal'
if data_type == 'coco':
    data_root_dir = './segmentation_data/coco_person_datasets'
    data_train = '%s/coco_train_%d,%d.rec' % (data_root_dir, resize_size[0], resize_size[1])
    data_val = '%s/coco_val_%d,%d.rec' % (data_root_dir, resize_size[0], resize_size[1])
elif data_type == 'portrait':
    data_root_dir = './segmentation_data/portrait_person_datasets'
    data_train = '%s/portrait_train_%d,%d.rec' % (data_root_dir, resize_size[0], resize_size[1])
    data_val = '%s/portrait_val_%d,%d.rec' % (data_root_dir, resize_size[0], resize_size[1])
elif data_type == 'pascal':
    data_root_dir = './segmentation_data/pascal_person_datasets'
    data_train = '%s/portrait_train_%d,%d.rec' % (data_root_dir, resize_size[0], resize_size[1])
    data_val = '%s/portrait_val_%d,%d.rec' % (data_root_dir, resize_size[0], resize_size[1])
elif data_type == 'lip':
    data_root_dir = './segmentation_data/lip_person_datasets'
    data_train = '%s/lip_train_%d,%d.rec' % (data_root_dir, resize_size[0], resize_size[1])
    data_val = '%s/lip_val_%d,%d.rec' % (data_root_dir, resize_size[0], resize_size[1])
elif data_type == 'coco_portrait_pascal':
    train_data_name = 'coco_portrait_pascal'
    data_root_dir = './segmentation_data/merge_person_datasets'
    data_train = '%s/%s_train_%d,%d.rec' % (data_root_dir, train_data_name, resize_size[0], resize_size[1])

    val_data_name = 'coco'
    data_root_dir = './segmentation_data/coco_person_datasets'
    data_val = '%s/%s_val_%d,%d.rec' % (data_root_dir, val_data_name, resize_size[0], resize_size[1])
else:
    raise NotImplementedError

# acc, meanIOU
eval_metric = 'meanIOU'
to_eval_train = True
# endregion


def main():
    # region 0. 准备模型
    # 旧模型，用于pre train 模型
    old_model_root_dir = './model_coco_person'
    old_batch_size = 8
    old_learning_rate = 1e-6
    pre_train_model_epoch = 50
    pre_train_model_type = 'FCN32s'
    pre_train_model_prefix = "%s/%s_VGG16_size%d_batch%d_lr%.0e" % (
        old_model_root_dir,
        pre_train_model_type,
        resize_size[0],
        old_batch_size,
        old_learning_rate,
    )
    # 新模型前缀
    # model_coco_person, model_pascal_person
    new_model_root_dir = './model_coco_person'
    new_batch_size = 8
    new_learning_rate = 1e-7
    fcnxs_new_model_prefix = "%s/%s_VGG16_data%s_size%d_batch%d_lr%.0e" % (
        new_model_root_dir,
        args.model,
        data_type,
        resize_size[0],
        new_batch_size,
        new_learning_rate,
    )
    begin_epoch = pre_train_model_epoch
    # if not continue_train:
    #     begin_epoch = 0
    logging.info('model prefix: %s' % fcnxs_new_model_prefix)
    logging.info('new_learning_rate: %.0e' % new_learning_rate)
    logging.info('batch_size: %d' % new_batch_size)
    logging.info('resize_size: %s' % str(resize_size))
    # endregion

    # if not continue_train:
    # region 1. 构建模型
    if args.model == 'FCN32s':
        fcnxs = symbol_fcnxs.get_fcn32s_symbol(numclass=2, workspace_default=2048)
    elif args.model == "FCN16s":
        fcnxs = symbol_fcnxs.get_fcn16s_symbol(numclass=2, workspace_default=1536)
    elif args.model == "FCN8s":
        fcnxs = symbol_fcnxs.get_fcn8s_symbol(numclass=2, workspace_default=1536)
    elif args.model == "FCN4s":
        fcnxs = symbol_fcnxs.get_fcn4s_symbol(numclass=2, workspace_default=1536)
    elif args.model == "FCN_atrous":
        fcnxs = symbol_fcnxs_atrous_person.get_fcnatrous_symbol(numclass=2, workspace_default=1536)
    else:
        raise NotImplementedError
    # endregion

    # region 2. 加载 pre-trained 的VGG16模型 并初始化 FCN模型
    logging.info('pre train with %s---000%d' % (pre_train_model_prefix, pre_train_model_epoch))
    _, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(pre_train_model_prefix, pre_train_model_epoch)
    # print fcnxs_args.keys()
    # pre train FCN模型
    if args.init_type == "vgg16":
        fcnxs_args, fcnxs_auxs = init_fcnxs.init_from_vgg16(ctx, fcnxs, fcnxs_args, fcnxs_auxs)
    elif args.init_type == "fcnxs":
        fcnxs_args, fcnxs_auxs = init_fcnxs.init_from_fcnxs(ctx, fcnxs, fcnxs_args, fcnxs_auxs)
    else:
        raise NotImplementedError
    # endregion
    # region 准备训练和验证数据
    train_dataiter, val_dataiter = get_train_val_iter(
        use_record_data=use_record_data,
        root_dir=data_root_dir,
        resize_size=resize_size,
        batch_size=new_batch_size,
        # cut_off_size         = 400,
        rgb_mean=(123.68, 116.779, 103.939),
        buffer_image_set=buffer_image_set,
        args={
            'data_train': data_train,
            'data_val': data_val,
            'image_shape': (3, resize_size[0], resize_size[1]),
            'rgb_mean': (123.68, 116.779, 103.939),
            'batch_size': new_batch_size,
            'data_nthreads': 50
        }
    )
    # quit()
    # endregion
    # region 开始训练
    model = Solver(
        ctx=ctx,
        symbol=fcnxs,
        begin_epoch=begin_epoch,
        num_epoch=args.epoch,
        arg_params=fcnxs_args,
        aux_params=fcnxs_auxs,
        learning_rate=new_learning_rate,
        momentum=0.99,
        wd=0.0005)
    model.fit(
        train_data=train_dataiter,
        eval_data=val_dataiter,
        period=period,
        to_eval_train=to_eval_train,
        eval_metric=eval_metric,
        batch_end_callback=mx.callback.Speedometer(batch_size=new_batch_size,
                                                   frequent=speedometer_freq,
                                                   auto_reset=False),
        epoch_end_callback=mx.callback.do_checkpoint(fcnxs_new_model_prefix, period=5)
    )
    # endregion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert vgg16 model to vgg16fc model.')
    parser.add_argument('--model', default='fcnxs',
                        help='The type of fcn-xs model, e.g. fcnxs, fcn16s, fcn8s.')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='gpus to run, e.g. 0 or 1. -1 means using cpu')
    parser.add_argument('--prefix', default='VGG_FC_ILSVRC_16_layers',
                        help='The prefix(include path) of vgg16 model with mxnet format.')
    parser.add_argument('--epoch', type=int, default=74,
                        help='The epoch number of vgg16 model.')
    parser.add_argument('--init-type', default="vgg16",
                        help='the init type of fcn-xs model, e.g. vgg16, fcnxs')
    parser.add_argument('--retrain', action='store_true', default=False,
                        help='true means continue training.')
    args = parser.parse_args()
    logging.info(args)
    if args.gpu == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)
    # ctx = mx.gpu(3)
    main()
