# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../python"))
import mxnet as mx
import random
import argparse
import cv2
import time
import traceback
import numpy as np

if sys.version_info[0] == 3:
    xrange = range

try:
    import multiprocessing
except ImportError:
    multiprocessing = None


def list_image(root, recursive, exts):
    i = 0
    if recursive:
        cat = {}
        for path, dirs, files in os.walk(root, followlinks=True):
            dirs.sort()
            files.sort()
            for fname in files:
                fpath = os.path.join(path, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and (suffix in exts):
                    if path not in cat:
                        cat[path] = len(cat)
                    yield (i, os.path.relpath(fpath, root), cat[path])
                    i += 1
        for k, v in sorted(cat.items(), key=lambda x: x[1]):
            print(os.path.relpath(k, root), v)
    else:
        for fname in sorted(os.listdir(root)):
            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                yield (i, os.path.relpath(fpath, root), 0)
                i += 1


def write_list(path_out, image_list):
    """ by WJD: 修改成可以处理多维度的标签, 直接把标签文件名保存

    :param path_out:
    :param image_list:
    :return:
    """
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            sys.stderr.write('\r第%d张图片：%s' % (i, str(item)))
            # print(item)
            # quit()
            # label = np.fromfile(os.path.join(args.root, '../masks/%s.bin' % item[1]), dtype=bool).astype(int)

            # print(label.shape)
            # print(label)
            # label = label.reshape(*img.shape[:2])
            # quit()
            line = '%d\t' % item[0]
            if args.data_type == 'test':
                for j in item[2:]:
                    line += '%f\t' % j
            else:
                line += '%s\t' % (label_path % item[1])
            line += '%s\n' % item[1]
            fout.write(line)


def make_list(args):
    image_list = list_image(args.root, args.recursive, args.exts)
    image_list = list(image_list)
    if args.shuffle is True:
        random.seed(100)
        random.shuffle(image_list)
    N = len(image_list)
    chunk_size = (N + args.chunks - 1) / args.chunks
    for i in xrange(args.chunks):
        chunk = image_list[i * chunk_size:(i + 1) * chunk_size]
        if args.chunks > 1:
            str_chunk = '_%d' % i
        else:
            str_chunk = ''
        sep = int(chunk_size * args.train_ratio)
        sep_test = int(chunk_size * args.test_ratio)
        if args.train_ratio == 1.0:
            write_list(args.prefix + str_chunk + '.lst', chunk)
        else:
            if args.test_ratio:
                write_list(args.prefix + str_chunk + '_test.lst', chunk[:sep_test])
            if args.train_ratio + args.test_ratio < 1.0:
                write_list(args.prefix + str_chunk + '_val.lst', chunk[sep_test + sep:])
            write_list(args.prefix + str_chunk + '_train.lst', chunk[sep_test:sep_test + sep])


def read_list(path_in):
    """ by WJD: 修改成可以处理多维度的标签

   :param path_in:
   :return:
   """
    with open(path_in) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split('\t')]
            line_len = len(line)
            if line_len < 3:
                print('lst should at least has three parts, but only has %s parts for %s' % (line_len, line))
                continue
            try:
                if args.data_type == 'test':
                    item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
                else:
                    item = [int(line[0])] + [line[-1]] + [str(i) for i in line[1:-1]]
            except Exception as e:
                print('Parsing lst met error for %s, detail: %s' % (line, e))
                continue
            yield item


def resize_img(img, newsize):
    return cv2.resize(img,
                      dsize=newsize,
                      interpolation=cv2.INTER_NEAREST,
                      )


def image_encode(args, i, item, q_out):
    """by WJD: 修改成兼容 多标签

    :param args:
    :param i:
    :param item:
    :param q_out:
    :return:
    """
    img_fullpath = os.path.join(args.root, item[1])
    label_fullpath = os.path.join(args.root, item[-1])
    # print(img_fullpath,label_fullpath)
    # print(item)
    label = np.fromfile(os.path.join(args.root, label_fullpath), dtype=label_dtype).astype(dtype=int)
    # quit()
    if args.pass_through:
        # 直接保存原图
        if len(item) > 3 and args.pack_label:
            header = mx.recordio.IRHeader(0, item[2:], item[0], 0)
        elif args.pack_label:
            # 输入是多标签的文件名
            header = mx.recordio.IRHeader(0, label, item[0], 0)
        else:
            print(item[2], item[0])
            header = mx.recordio.IRHeader(0, item[2], item[0], 0)
        try:
            with open(img_fullpath, 'rb') as fin:
                img = fin.read()
            s = mx.recordio.pack(header, img)
            q_out.put((i, s, item))
        except Exception as e:
            traceback.print_exc()
            print('pack_img error:', item[1], e)
            q_out.put((i, None, item))
        return
    # region 如果需要进行图片变换,才进入这里
    try:
        # print(img_fullpath)
        img = cv2.imread(img_fullpath, args.color)
        # print(img.shape)
        # print(label.shape)
        # quit()
        label = label.reshape(*img.shape[:2])
    except:
        traceback.print_exc()
        print('imread error trying to load file: %s ' % img_fullpath)
        q_out.put((i, None, item))
        return
    if img is None:
        print('imread read blank (None) image for file: %s' % img_fullpath)
        q_out.put((i, None, item))
        return
    if args.center_crop:
        if img.shape[0] > img.shape[1]:
            margin = (img.shape[0] - img.shape[1]) / 2
            img = img[margin:margin + img.shape[1], :]
        else:
            margin = (img.shape[1] - img.shape[0]) / 2
            img = img[:, margin:margin + img.shape[0]]

    if args.resize:
        if img.shape[0] > img.shape[1]:
            newsize = (args.resize, img.shape[0] * args.resize / img.shape[1])
        else:
            newsize = (img.shape[1] * args.resize / img.shape[0], args.resize)
        img = cv2.resize(img, newsize)

    if len(args.resize_to.strip()) > 0:
        # 将图片resize到特定大小
        newsize = tuple(map(int, args.resize_to.split(',')))
        # print self.resize_size
        img = resize_img(img, (newsize[1], newsize[0]))
        # newsize = (100, 75)
        label = resize_img(label, (newsize[1], newsize[0]))
        assert sum(label.flatten()[(label.flatten() != 1) * (label.flatten() != 0)]) == 0, 'label只能是0/1的数值！'

    try:
        # print(item, args.pack_label)
        if len(item) > 3 and args.pack_label:
            header = mx.recordio.IRHeader(0, item[2:], item[0], 0)
        elif args.pack_label:
            # 输入是多标签的文件名
            header = mx.recordio.IRHeader(0, label, item[0], 0)
        else:
            header = mx.recordio.IRHeader(0, item[2], item[0], 0)
        s = mx.recordio.pack_img(header, img, quality=args.quality, img_fmt=args.encoding)
        q_out.put((i, s, item))
    except Exception as e:
        traceback.print_exc()
        print('pack_img error on file: %s' % img_fullpath, e)
        q_out.put((i, None, item))
        return


def read_worker(args, q_in, q_out):
    while True:
        deq = q_in.get()
        if deq is None:
            break
        i, item = deq
        image_encode(args, i, item, q_out)


def write_worker(q_out, fname, working_dir):
    pre_time = time.time()
    count = 0
    fname = os.path.basename(fname)
    fname_rec = os.path.splitext(fname)[0] + '.rec'
    fname_idx = os.path.splitext(fname)[0] + '.idx'
    record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                           os.path.join(working_dir, fname_rec), 'w')
    buf = {}
    more = True
    while more:
        deq = q_out.get()
        if deq is not None:
            i, s, item = deq
            buf[i] = (s, item)
        else:
            more = False
        while count in buf:
            s, item = buf[count]
            del buf[count]
            if s is not None:
                record.write_idx(item[0], s)

            if count % 1000 == 0:
                cur_time = time.time()
                print('time:', cur_time - pre_time, ' count:', count)
                pre_time = cur_time
            count += 1


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    parser.add_argument('prefix', help='prefix of input/output lst and rec files.')
    parser.add_argument('root', help='path to folder containing images.')
    parser.add_argument('data_type', help='data_type.etc. coco, pascal, portrait.')

    cgroup = parser.add_argument_group('Options for creating image lists')
    cgroup.add_argument('--list', type=bool, default=False,
                        help='If this is set im2rec will create image list(s) by traversing root folder\
        and output to <prefix>.lst.\
        Otherwise im2rec will read <prefix>.lst and create a database at <prefix>.rec')
    cgroup.add_argument('--exts', nargs='+', default=['.jpeg', '.jpg'],
                        help='list of acceptable image extensions.')
    cgroup.add_argument('--chunks', type=int, default=1, help='number of chunks.')
    cgroup.add_argument('--train-ratio', type=float, default=1.0,
                        help='Ratio of images to use for training.')
    cgroup.add_argument('--test-ratio', type=float, default=0,
                        help='Ratio of images to use for testing.')
    cgroup.add_argument('--recursive', type=bool, default=False,
                        help='If true recursively walk through subdirs and assign an unique label\
        to images in each folder. Otherwise only include images in the root folder\
        and give them label 0.')
    cgroup.add_argument('--shuffle', type=bool, default=True, help='If this is set as True, \
        im2rec will randomize the image order in <prefix>.lst')

    rgroup = parser.add_argument_group('Options for creating database')
    rgroup.add_argument('--pass-through', type=bool, default=False,
                        help='whether to skip transformation and save image as is')
    rgroup.add_argument('--resize', type=int, default=0,
                        help='resize the shorter edge of image to the newsize, original images will\
        be packed by default.')
    rgroup.add_argument('--resize_to', type=str, default='',
                        help='resize the size of image to the newsize, original images will\
        be packed by default.')
    rgroup.add_argument('--center-crop', type=bool, default=False,
                        help='specify whether to crop the center image to make it rectangular.')
    rgroup.add_argument('--quality', type=int, default=95,
                        help='JPEG quality for encoding, 1-100; or PNG compression for encoding, 1-9')
    rgroup.add_argument('--num-thread', type=int, default=1,
                        help='number of thread to use for encoding. order of images will be different\
        from the input list if >1. the input list will be modified to match the\
        resulting order.')
    rgroup.add_argument('--color', type=int, default=1, choices=[-1, 0, 1],
                        help='specify the color mode of the loaded image.\
        1: Loads a color image. Any transparency of image will be neglected. It is the default flag.\
        0: Loads image in grayscale mode.\
        -1:Loads image as such including alpha channel.')
    rgroup.add_argument('--encoding', type=str, default='.jpg', choices=['.jpg', '.png'],
                        help='specify the encoding of the images.')
    rgroup.add_argument('--pack-label', type=int, default=0,
                        help='Whether to also pack multi dimensional label in the record file')
    args = parser.parse_args()
    args.prefix = os.path.abspath(args.prefix)
    args.root = os.path.abspath(args.root)
    return args


# python im2rec.py --list=True coco_train ~/Projects/fcn-xs/coco_person_datasets/images_train/ --num-thread=10
# python im2rec.py coco_train ~/Projects/fcn-xs/coco_person_datasets/images_train/
# --num-thread=10 --pack-label=True --resize_to='256,256' --quality=100
if __name__ == '__main__':
    args = parse_args()

    data_type = args.data_type
    if data_type == 'coco':
        # 标签文件夹位置（相对于 args.root的相对位置）
        label_path = '../masks/%s.bin'
        label_dtype = bool
    elif data_type == 'pascal':
        label_path = '%s'
        label_dtype = np.int64
    elif data_type == 'portrait':
        label_path = '../masks/%s.bin'
        label_dtype = np.uint8
    elif data_type == 'test':
        # 这个模式下是不需要 标签 的, 直接
        pass
    else:
        raise NotImplementedError

    if args.list:
        make_list(args)
    else:
        if os.path.isdir(args.prefix):
            working_dir = args.prefix
        else:
            working_dir = os.path.dirname(args.prefix)
        files = [os.path.join(working_dir, fname) for fname in os.listdir(working_dir)
                 if os.path.isfile(os.path.join(working_dir, fname))]
        count = 0
        for fname in files:
            if fname.startswith(args.prefix) and fname.endswith('.lst'):
                print('Creating .rec file from', fname, 'in', working_dir)
                count += 1
                image_list = read_list(fname)
                # -- write_record -- #
                if args.num_thread > 1 and multiprocessing is not None:
                    q_in = [multiprocessing.Queue(1024) for i in range(args.num_thread)]
                    q_out = multiprocessing.Queue(1024)
                    read_process = [multiprocessing.Process(target=read_worker, args=(args, q_in[i], q_out)) \
                                    for i in range(args.num_thread)]
                    for p in read_process:
                        p.start()
                    write_process = multiprocessing.Process(target=write_worker, args=(q_out, fname, working_dir))
                    write_process.start()

                    for i, item in enumerate(image_list):
                        q_in[i % len(q_in)].put((i, item))
                    for q in q_in:
                        q.put(None)
                    for p in read_process:
                        p.join()

                    q_out.put(None)
                    write_process.join()
                else:
                    print('multiprocessing not available, fall back to single threaded encoding')
                    try:
                        import Queue as queue
                    except ImportError:
                        import queue
                    q_out = queue.Queue()
                    fname = os.path.basename(fname)
                    fname_rec = os.path.splitext(fname)[0] + '.rec'
                    fname_idx = os.path.splitext(fname)[0] + '.idx'
                    record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                                           os.path.join(working_dir, fname_rec), 'w')
                    cnt = 0
                    pre_time = time.time()
                    for i, item in enumerate(image_list):
                        image_encode(args, i, item, q_out)
                        if q_out.empty():
                            continue
                        _, s, _ = q_out.get()
                        record.write_idx(item[0], s)
                        if cnt % 1000 == 0:
                            cur_time = time.time()
                            print('time:', cur_time - pre_time, ' count:', cnt)
                            pre_time = cur_time
                        cnt += 1
        if not count:
            print('Did not find and list file with prefix %s' % args.prefix)
