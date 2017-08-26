# encoding=utf8
from symbol_fcnxs_person import get_fcn32s_symbol, offset
import mxnet as mx
import logging
from PIL import Image
from skimage import io, transform
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from metric_mIOU import MeanIoU

# batch_size = 10
# size = resize_size
size = (256, 256)
batch_size = 8
rgb_mean = (123.68, 116.779, 103.939)
# path_imgrec="./coco_person_datasets/coco_val_%d,%d.rec"%(size[0],size[1])
# path_imgrec="./pascal_voc_person_datasets/pascal_val_%d,%d.rec" % (size[0], size[1])
# path_imgrec = "/home/wangjundong/data/segmentation_data/lip_person_datasets/lip_train_%d,%d.rec" % (size[0], size[1])
train_path_imgrec = "/home/wangjundong/data/segmentation_data/portrait_person_datasets/portrait_train_%d,%d.rec" % (size[0], size[1])
val_path_imgrec = "/home/wangjundong/data/segmentation_data/portrait_person_datasets/portrait_val_%d,%d.rec" % (size[0], size[1])
# path_imglist="./potrait_person_datasets/portrait_val.lst"
# path_imgrec="/home/wangjundong/mxnet/tools/pp_val.rec"
# path_imglist="/home/wangjundong/mxnet/tools/pp_val.lst"
dataiter1 = mx.io.ImageRecordIter_v1(
    # rec文件的路径
    #     path_imglist=path_imglist,
    path_imgrec=train_path_imgrec,
    # iterator生成的每一个实例的shape
    data_shape=(3, size[0], size[1]),
    mean_r=rgb_mean[0],
    mean_g=rgb_mean[1],
    mean_b=rgb_mean[2],
    data_name='data',
    label_name='softmax_label',
    # batch的大小
    batch_size=batch_size,
    # 是否随机从原图中切取出一个data_shape大小
    rand_crop=False,
    # 是否随机水平反射
    rand_mirror=False,
    label_width=size[0] * size[1],

)
dataiter2 = mx.io.ImageRecordIter_v1(
    # rec文件的路径
    path_imgrec=val_path_imgrec,
    # iterator生成的每一个实例的shape
    data_shape=(3, size[0], size[1]),
    mean_r=rgb_mean[0],
    mean_g=rgb_mean[1],
    mean_b=rgb_mean[2],
    data_name='data',
    label_name='softmax_label',
    # batch的大小
    batch_size=batch_size,
    # 是否随机从原图中切取出一个data_shape大小
    rand_crop=False,
    # 是否随机水平反射

    rand_mirror=False,
    label_width=size[0] * size[1]
)

fcn32s = get_fcn32s_symbol(
    numclass=2
)
mod = mx.mod.Module(fcn32s, context=mx.gpu(0))
mod.bind(data_shapes=dataiter1.provide_data,
         label_shapes=dataiter1.provide_label)
mod.init_params()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
mod.fit(dataiter1,dataiter2,
        num_epoch=10,
        eval_metric=MeanIoU(c=1,),
        # eval_metric='acc',
        batch_end_callback=mx.callback.Speedometer(batch_size=dataiter1.batch_size, frequent=50),
        )
