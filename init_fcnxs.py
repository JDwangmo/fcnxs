# encoding=utf8
# pylint: skip-file
import mxnet as mx
import numpy as np
import sys
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# make a bilinear interpolation kernel, return a numpy.ndarray
def upsample_filt(size):  # 64
    factor = (size + 1) // 2
    # print factor # 32
    if size % 2 == 1:
        center = factor - 1.0
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    # print og
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def init_from_vgg16(ctx, fcnxs_symbol, vgg16fc_args, vgg16fc_auxs):
    fcnxs_args = vgg16fc_args.copy()
    fcnxs_auxs = vgg16fc_auxs.copy()
    # print len(fcnxs_args)
    # print len(fcnxs_symbol.list_arguments())
    for k, v in fcnxs_args.items():
        if v.context != ctx:
            fcnxs_args[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcnxs_args[k])
    for k, v in fcnxs_auxs.items():
        if v.context != ctx:
            fcnxs_auxs[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcnxs_auxs[k])
    data_shape = (1, 3, 500, 500)
    arg_names = fcnxs_symbol.list_arguments()
    arg_shapes, _, _ = fcnxs_symbol.infer_shape(data=data_shape)
    rest_params = dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
                        if x[0] in ['score_weight', 'score_bias', 'score_pool4_weight', 'score_pool4_bias',
                                    'score_pool3_weight', 'score_pool3_bias']])
    fcnxs_args.update(rest_params)
    # print fcnxs_args.keys()

    deconv_params = dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes)
                          if x[0] in ["bigscore_weight", 'score2_weight', 'score4_weight']])
    for k, v in deconv_params.items():
        # print k, v
        # print v[3]
        filt = upsample_filt(v[3])
        # print filt
        # quit()
        initw = np.zeros(v)
        initw[range(v[0]), range(v[1]), :, :] = filt  # becareful here is the slice assing
        fcnxs_args[k] = mx.nd.array(initw, ctx)
    # print len(fcnxs_args)
    # quit()

    return fcnxs_args, fcnxs_auxs


def init_from_fcnxs(ctx, fcnxs_symbol, fcnxs_args_from, fcnxs_auxs_from):
    """ use zero initialization for better convergence, because it tends to output 0,
    and the label 0 stands for background, which may occupy most size of one image.
    """
    fcnxs_args = fcnxs_args_from.copy()
    fcnxs_auxs = fcnxs_auxs_from.copy()
    for k, v in fcnxs_args.items():
        if v.context != ctx:
            fcnxs_args[k] = mx.nd.empty(v.shape, ctx)
            v.copyto(fcnxs_args[k])
    for k, v in fcnxs_auxs.items():
        if v.context != ctx:
            fcnxs_auxs[k] = mx.nd.empty(v.shape, ctx)
            v.copyto(fcnxs_auxs[k])
    data_shape = (1, 3, 500, 500)
    arg_names = fcnxs_symbol.list_arguments()
    arg_shapes, _, _ = fcnxs_symbol.infer_shape(data=data_shape)
    rest_params = {}
    deconv_params = {}
    if 'score_pool3_weight' in arg_names:
        # this is fcn8s init from fcn16s
        if 'score_pool3_weight' in fcnxs_args_from:
            logging.info("score_pool3_weight使用原模型参数继续训练!")
        else:
            rest_params.update(dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
                                     if x[0] in ['score_pool3_bias', 'score_pool3_weight']]))
            deconv_params.update(dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes) if x[0]
                                       in ["bigscore_weight", 'score4_weight']]))
    if 'score_pool4_weight' in arg_names:
        # this is fcn16s init from fcn32s
        if 'score_pool4_weight' in fcnxs_args_from:
            logging.info("score_pool4_weight使用原模型参数继续训练!")
        else:
            # 初始化参数
            rest_params.update(dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
                                     if x[0] in ['score_pool4_weight', 'score_pool4_bias']]))
            deconv_params.update(dict(
                [(x[0], x[1]) for x in zip(arg_names, arg_shapes) if x[0] in ["bigscore_weight", 'score2_weight']]))
            # else:
            #     this is fcn32s init
            # rest_params = {}
            # deconv_params = {}
            # logging.info("使用原模型参数继续训练!")
            # logging.error("you are init the fcn32s model, so you should use init_from_vgg16()")
            # sys.exit()

    if 'ct_conv1_1_weight' in arg_names:
        # this is FCN_atrous
        if 'ct_conv1_1_weight' in fcnxs_args_from:
            logging.info("score_pool4_weight使用原模型参数继续训练!")
        else:
            # 初始化参数
            rest_params.update(dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
                                     if x[0] not in fcnxs_args_from]))
            deconv_params.update(dict(
                [(x[0], x[1]) for x in zip(arg_names, arg_shapes) if x[0] in ["bigscore_weight", 'score2_weight']]))
            # else:
            #     this is fcn32s init
            # rest_params = {}
            # deconv_params = {}
            # logging.info("使用原模型参数继续训练!")
            # logging.error("you are init the fcn32s model, so you should use init_from_vgg16()")
            # sys.exit()

    if 'score8_weight' in arg_names:
        # this is FCN_4s
        if 'score8_weight' in fcnxs_args_from:
            logging.info("score8_weight使用原模型参数继续训练!")
        else:
            # 初始化参数
            rest_params.update(dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
                                     if x[0] not in fcnxs_args_from]))
            deconv_params.update(dict(
                [(x[0], x[1]) for x in zip(arg_names, arg_shapes) if
                 x[0] in ["bigscore_weight", 'score2_weight', 'score4_weight', 'score8_weight']]))
            # else:
            #     this is fcn32s init
            # rest_params = {}
            # deconv_params = {}
            # logging.info("使用原模型参数继续训练!")
            # logging.error("you are init the fcn32s model, so you should use init_from_vgg16()")
            # sys.exit()
    # print deconv_params
    # quit()
    for x in zip(arg_names, arg_shapes):
        # 检测是否有出现跟原模型参数 shape不一致的情况
        if x[0] in fcnxs_args_from and fcnxs_args_from[x[0]].shape != x[1] and x[0] not in ['data', 'softmax_label']:
            if x[0] in ["bigscore_weight", 'score2_weight', 'score4_weight']:
                deconv_params[x[0]] = x[1]
                logging.info('参数形状不一致！%s,%s,%s,反卷积参数重新设置！' % (x[0], str(fcnxs_args_from[x[0]].shape), x[1]))
            else:
                logging.info('参数形状不一致！%s,%s,%s' % (x[0], str(fcnxs_args[x[0]].shape), x[1]))
    fcnxs_args.update(rest_params)
    for k, v in deconv_params.items():
        filt = upsample_filt(v[3])
        initw = np.zeros(v)
        initw[range(v[0]), range(v[1]), :, :] = filt  # becareful here is the slice assing
        fcnxs_args[k] = mx.nd.array(initw, ctx)
    return fcnxs_args, fcnxs_auxs
