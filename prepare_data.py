# encoding=utf8

from data_person import FileIter
import sys, os
import argparse
import mxnet as mx
import numpy as np
import logging
from data_person import FileIter

data_root_dir = './coco_person_datasets'
resize_size = (128, 128)
batch_size = 64


def main():
    # region 准备训练和验证数据
    train_dataiter = FileIter(
        data_type='train',
        root_dir=data_root_dir,
        flist_name="train.txt",
        resize_size=resize_size,
        batch_size=batch_size,
        # cut_off_size         = 400,
        rgb_mean=(123.68, 116.779, 103.939),
        buffer_image_set=True
    )
    val_dataiter = FileIter(
        data_type='val',
        root_dir=data_root_dir,
        flist_name="val.txt",
        batch_size=batch_size,
        resize_size=resize_size,
        rgb_mean=(123.68, 116.779, 103.939),
        buffer_image_set=True
    )
    # endregion


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Prepare data for FCN model.')
    #
    # parser.add_argument('--dataset', default='coco',
    #                     help='data set, e.g.coco,.')
    # parser.add_argument('--resize', default=(256, 256),
    #                     help='resize image to specific size')
    # parser.add_argument('--destination dir', default='./processed_data',
    #                     help='save to where')
    #
    # args = parser.parse_args()
    # logging.info(args)
    main()
