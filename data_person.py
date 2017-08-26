# encoding=utf8
# pylint: skip-file
""" file iterator for pasval voc 2012"""
import mxnet as mx
import numpy as np
import sys, os
from mxnet.io import DataIter, DataBatch
from PIL import Image
from skimage import transform
from skimage import io
import logging
import cv2
import pickle
import time
import pandas as pd


class FileIter(DataIter):
    """FileIter object in fcn-xs example. Taking a file list file to get dataiter.
    in this example, we use the whole image training for fcn-xs, that is to say
    we do not need resize/crop the image to the same size, so the batch_size is
    set to 1 here
    Parameters
    ----------
    root_dir : string
        the root dir of image/label lie in
    flist_name : string
        the list file of iamge and label, every line owns the form:
        index \t image_data_path \t image_label_path
    cut_off_size : int
        if the maximal size of one image is larger than cut_off_size, then it will
        crop the image with the minimal size of that image
    data_name : string
        the data name used in symbol data(default data name)
    label_name : string
        the label name used in symbol softmax_label(default label name)
    """

    def __init__(self,
                 data_type='',
                 root_dir=None,
                 flist_name=None,
                 batch_size=1,
                 rgb_mean=(117, 117, 117),
                 resize_size=None,
                 cut_off_size=None,
                 data_name="data",
                 label_name="softmax_label",
                 buffer_image_set=True,
                 logger=None,
                 ):
        """

        :type buffer_image_set: bool
        :param root_dir:
        :param flist_name:
        :param rgb_mean:
        :param resize_size:
        :param cut_off_size:
        :param data_name:
        :param label_name:
        """
        super(FileIter, self).__init__(batch_size=batch_size)
        self.batch_image_name, self.batch_label_name = None, None
        self.batch_image, self.batch_label, self.batch_img_size, self.batch_original_img_size = None, None, None, None
        if logger is None:
            logging.basicConfig()
            logger = logging.getLogger('DataIter')
            logger.setLevel(logging.INFO)
            self.logger = logger
        else:
            self.logger = logger
        self.data_type = data_type
        self.root_dir = root_dir
        self.flist_name = os.path.join(self.root_dir, flist_name)

        self.mean = np.array(rgb_mean)  # (R, G, B)
        self.cut_off_size = cut_off_size
        self.resize_size = resize_size
        self.data_name = data_name
        self.label_name = label_name

        images_name = [line.strip() for line in open(self.flist_name, 'r').readlines()]
        self.num_data = len(images_name)
        # 加载图片和mask路径
        self.images_path = ['images/%s' % name for name in images_name]
        self.labels_path = ['masks/%s.bin' % name for name in images_name]
        # print self.labels_path[:10]

        self.logger.info('数据集大小：%d' % self.num_data)
        self.logger.info('参数设置：\n\tbatch_size:%s\n' % (str(self.batch_size)))
        self.logger.info('预处理参数：\n\tresize_size:%s\n' % (str(self.resize_size)))

        self.cursor = 0
        # 图片集
        self.image_set = None
        self.label_set = None
        self.original_img_size_set = None
        if buffer_image_set:
            self.load_image_set()

    def load_image_set(self):
        """ 一次性读取所有图片集

        :return:
        """
        buffer_data_path = 'processed_data/DATA%s_TYPE%s_RESIZE%s.data' % (
            self.root_dir.split('/')[-1],
            self.data_type,
            str(self.resize_size)
        )
        if os.path.exists(buffer_data_path):
            self.logger.info('正在加载缓存数据：%s\n' % buffer_data_path)
            image_set = np.fromfile(buffer_data_path)
            label_set = np.fromfile('%s.label' % buffer_data_path, dtype=np.int32)
            original_img_size_set = np.fromfile('%s.ori_size' % buffer_data_path, dtype=np.int64)

            self.image_set = image_set.reshape(-1, 3, self.resize_size[0], self.resize_size[1])
            self.label_set = label_set.reshape(-1, self.resize_size[0], self.resize_size[1])
            self.original_img_size_set = original_img_size_set.reshape(-1, 3)
            print self.image_set.shape
            print self.label_set.shape
            print self.original_img_size_set.shape
        else:
            self.logger.info('正在缓存数据：%s\n' % buffer_data_path)
            self.image_set, self.label_set, self.original_img_size_set = [], [], []
            for index, (img_path, label_path) in enumerate(zip(self.images_path, self.labels_path)):
                sys.stderr.write('\r正在处理第%d/%d张图片:%s' % (index, self.num_data, img_path))
                # if index > 10:
                #     break
                img, label, original_img_size = self._read_img(img_path, label_path)

                self.image_set.append(img)
                self.label_set.append(label)
                self.original_img_size_set.append(list(original_img_size))
            image_set = np.concatenate(self.image_set)
            label_set = np.concatenate(self.label_set)
            original_img_size_set = np.array(self.original_img_size_set)
            # print image_set.shape
            # print label_set.shape
            # print original_img_size_set.shape
            # print original_img_size_set
            image_set.tofile(buffer_data_path)
            label_set.tofile('%s.label' % buffer_data_path)
            original_img_size_set.tofile('%s.ori_size' % buffer_data_path)
            # print sum(label_set.flatten() == 1), sum(label_set.flatten() == 0)
            # print original_img_size_set.dtype
            # print original_img_size_set.shape
            # original_img_size_set = np.fromfile('%s.ori_size' % buffer_data_path,dtype=np.int64)
            # print original_img_size_set.dtype
            # print original_img_size_set.shape
            # print sum(label_set.flatten() == 1), sum(label_set.flatten() == 0)
            # quit()
            # pickle.dump(self.image_set, open(buffer_data_path, 'wb'))
            # pickle.dump(self.label_set, open('%s.label' % buffer_data_path, 'wb'))
            # pickle.dump(self.original_img_size_set, open('%s.ori_size' % buffer_data_path, 'wb'))
            sys.stderr.write('\n')
        self.logger.info('数据缓存/加载完毕：%d条数据' % len(self.image_set))
        self.num_data = len(self.image_set)

    def get_img_rgb_mean_std(self):
        """计算图片的均值核方差

        :param root_dir:
        :param flist_name:
        :return:
        """
        imgs = []
        for index, path in enumerate(self.images_path):
            sys.stderr.write('\r第%d张图片:%s' % (index, path))
            img = cv2.imread(os.path.join(self.root_dir, path))
            img = self.resize_img(img)
            imgs.append(img)
        mean = np.mean(imgs, axis=0)
        std = np.std(imgs, axis=0)
        return mean, std

    def getdata(self):
        """Get data of current batch.

        Returns
        -------
        list of NDArray
            The data of the current batch.
        """
        batch_image, batch_label, batch_img_size, batch_original_img_size, batch_image_name, batch_label_name \
            = [], [], [], [], [], []
        # self.cursor = 40000
        # start = time.time()
        for i in range(self.batch_size):
            if self.cursor + i <= self.num_data - 1:
                img_index = self.cursor + i
                # img_path, label_path = self.images_path[self.cursor + i], self.labels_path[self.cursor + i]
            else:
                img_index = i
            # start1 = time.time()
            img, label, original_img_size = self._read(img_index)
            # end1 = time.time()
            # print 'read: %f' % (end1 - start1)
            img_path, label_path = self.images_path[img_index], self.labels_path[img_index]
            batch_image.append(img)
            batch_label.append(label.reshape(1, -1))
            batch_img_size.append(self.resize_size)
            batch_original_img_size.append(original_img_size)
            batch_image_name.append(img_path)
            batch_label_name.append(label_path)
        batch_image = np.concatenate(batch_image)
        batch_label = np.concatenate(batch_label)
        # print batch_image.shape
        # end = time.time()
        # print 'getdata: %f' % (end - start)
        # quit()
        return batch_image, batch_label, batch_img_size, batch_original_img_size, batch_image_name, batch_label_name

    def resize_img(self, img):
        return cv2.resize(img,
                          dsize=self.resize_size,
                          interpolation=cv2.INTER_NEAREST,
                          )

    def _read(self, img_index):
        """ 返回图片和标签

        :param img_index:
        :return:
        """
        if self.image_set is None:
            img_path, label_path = self.images_path[img_index], self.labels_path[img_index]
            return self._read_img(img_path, label_path)
        else:
            # print self.images_path.index(img_path), img_path
            # print self.image_set.shape
            # quit()
            return np.expand_dims(self.image_set[img_index], axis=0), \
                   np.expand_dims(self.label_set[img_index], axis=0), \
                   self.original_img_size_set[img_index]

    def _read_img(self, img_path, label_path):
        """ 给定图片和标签的文件名，返回图片和标签数组

        :param img_path:
        :param label_path:
        :return: (np.array, np.array, ())
        """
        # region 读取图片
        # img = Image.open(os.path.join(self.root_dir, img_path))
        # (h,w,c)
        # start_all = time.time()
        # start = time.time()
        img = cv2.imread(os.path.join(self.root_dir, img_path), flags=1)
        img = np.array(img, dtype=np.float32)  # (h, w, c)
        original_img_size = img.shape
        # end = time.time()
        # print 'open img time: %f' % (end - start)
        # print img.shape
        # print img_path
        # (h, w)
        # start = time.time()
        # label = np.loadtxt(os.path.join(self.root_dir, label_path), delimiter=',')
        # label = pd.read_csv(os.path.join(self.root_dir, label_path), sep=',')
        label = np.fromfile(os.path.join(self.root_dir, label_path), dtype=bool).astype(int)
        # print label.shape
        label = label.reshape(*img.shape[:2])
        # end = time.time()
        # print 'load label time: %f' % (end - start)
        # quit()
        assert img.shape[:2] == label.shape
        # quit()
        # print img.shape
        # print label.shape
        # print img
        # label = np.array(label)  # (h, w)
        # endregion
        # start = time.time()
        # region resize操作
        if self.resize_size is not None:
            assert isinstance(self.resize_size, tuple) and len(self.resize_size) == 2, 'resize要为tuple类型！'
            # print self.resize_size
            img = self.resize_img(img)
            label = self.resize_img(label)
            assert sum(label.flatten()[(label.flatten() != 1) * (label.flatten() != 0)]) == 0, 'label只能是0/1的数值！'
        # end = time.time()
        # print 'resize time: %f' % (end - start)
        # endregion

        # region zero centered
        reshaped_mean = self.mean.reshape(1, 1, 3)
        img = img - reshaped_mean
        # endregion

        # region cut_off_size
        if self.cut_off_size is not None:
            max_hw = max(img.shape[0], img.shape[1])
            min_hw = min(img.shape[0], img.shape[1])
            if min_hw > self.cut_off_size:
                rand_start_max = int(np.random.uniform(0, max_hw - self.cut_off_size - 1))
                rand_start_min = int(np.random.uniform(0, min_hw - self.cut_off_size - 1))
                if img.shape[0] == max_hw:
                    img = img[rand_start_max: rand_start_max + self.cut_off_size,
                          rand_start_min: rand_start_min + self.cut_off_size]
                    label = label[rand_start_max: rand_start_max + self.cut_off_size,
                            rand_start_min: rand_start_min + self.cut_off_size]
                else:
                    img = img[rand_start_min: rand_start_min + self.cut_off_size,
                          rand_start_max: rand_start_max + self.cut_off_size]
                    label = label[rand_start_min: rand_start_min + self.cut_off_size,
                            rand_start_max: rand_start_max + self.cut_off_size]
            elif max_hw > self.cut_off_size:
                rand_start = int(np.random.uniform(0, max_hw - min_hw - 1))
                if img.shape[0] == max_hw:
                    img = img[rand_start: rand_start + min_hw, :]
                    label = label[rand_start: rand_start + min_hw, :]
                else:
                    img = img[:, rand_start: rand_start + min_hw]
                    label = label[:, rand_start: rand_start + min_hw]
        # endregion
        # (c, h, w)
        img = img.transpose((2, 0, 1))
        # img = np.swapaxes(img, 0, 2)  # (c, w, h)
        # img = np.swapaxes(img, 1, 2)  # (c, h, w)
        img = np.expand_dims(img, axis=0)  # (1, c, h, w)
        # label = np.array(label)  # (h, w)
        label = np.expand_dims(label, axis=0)  # (1, h, w)
        # print img.shape, label.shape
        # quit()
        # img = mx.nd.array(img)
        # label = mx.nd.array(label)
        # end_all = time.time()
        # print 'all time: %f' % (end_all - start_all)
        return img, label, original_img_size

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        # print self.data[0][1].shape
        # quit()
        # print self.batch_img_size
        # quit()
        # return [(k, tuple([1] + list(v.shape[1:]))) for k, v in self.data]
        return [mx.io.DataDesc(self.data_name, (self.batch_size, 3, self.resize_size[0], self.resize_size[1]))]
        # return [(k, v) for k, v in zip(self.batch_image_name, self.batch_img_size)]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        # return [(k, tuple([1] + list(v.shape[1:]))) for k, v in self.label]
        return [mx.io.DataDesc(self.label_name, (self.batch_size, self.resize_size[0] * self.resize_size[1]))]
        # return [(k, v) for k, v in zip(self.batch_label_name, self.batch_img_size)]

    # def get_batch_size(self):
    #     return 1

    def reset(self):
        self.cursor = 0
        # self.f.close()
        # self.f = open(self.flist_name, 'r')

    def iter_next(self):
        # self.cursor += 1
        if self.cursor <= self.num_data - 1:
            return True
        else:
            return False

    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            # self.data, self.label,\
            start = time.time()
            self.batch_image, self.batch_label, self.batch_img_size, self.batch_original_img_size, \
            self.batch_image_name, self.batch_label_name = self.getdata()
            # end = time.time()
            # print 'load data: %s' % (end - start)
            # print self.batch_image_name
            self.cursor += self.batch_size
            # return DataBatch(self.batch_image,
            #                  label=self.batch_label,
            #                  )
            # print mx.nd.concatenate(self.batch_image)
            # print self.batch_label
            # return {
            #     self.data_name: mx.nd.array(self.batch_image),
            #     self.label_name: mx.nd.array(self.batch_label),
            # }
            return DataBatch(data=[mx.nd.array(self.batch_image)],
                             label=[mx.nd.array(self.batch_label)],
                             )
        else:
            raise StopIteration


def get_train_val_iter(
        use_record_data=True,
        root_dir=None,
        resize_size=None,
        batch_size=1,
        # cut_off_size         = 400,
        rgb_mean=(123.68, 116.779, 103.939),
        buffer_image_set=True,
        args=None
):
    if not use_record_data:
        # region 准备训练和验证数据
        train = FileIter(
            data_type='train',
            root_dir=root_dir,
            flist_name="train.txt",
            resize_size=resize_size,
            batch_size=batch_size,
            # cut_off_size         = 400,
            rgb_mean=rgb_mean,
            buffer_image_set=buffer_image_set,
        )
        val = FileIter(
            data_type='val',
            root_dir=root_dir,
            flist_name="val.txt",
            batch_size=batch_size,
            resize_size=resize_size,
            rgb_mean=rgb_mean,
            buffer_image_set=buffer_image_set,
        )
        # endregion
        return train, val
    else:
        return get_rec_iter(args)


def get_rec_iter(args):
    image_shape = args['image_shape']

    rgb_mean = args['rgb_mean']

    train = mx.io.ImageRecordIter(
        path_imgrec=args['data_train'],
        label_width=image_shape[1] * image_shape[2],
        mean_r=rgb_mean[0],
        mean_g=rgb_mean[1],
        mean_b=rgb_mean[2],
        data_name='data',
        label_name='softmax_label',
        data_shape=image_shape,
        batch_size=args['batch_size'],
        preprocess_threads=args['data_nthreads'],
        shuffle=True,
    )

    if 'data_val' not in args:
        return train, None
    # quit()
    val = mx.io.ImageRecordIter(
        path_imgrec=args['data_val'],
        label_width=image_shape[1] * image_shape[2],
        mean_r=rgb_mean[0],
        mean_g=rgb_mean[1],
        mean_b=rgb_mean[2],
        data_name='data',
        label_name='softmax_label',
        batch_size=args['batch_size'],
        data_shape=image_shape,
        preprocess_threads=args['data_nthreads'],
    )

    return train, val
