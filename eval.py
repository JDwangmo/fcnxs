# encoding=utf8
# pylint: skip-file
import numpy as np
import argparse
import logging
import time
import mxnet as mx
from PIL import Image
from matplotlib import pyplot as plt
from mxnet import metric
from metric_mIOU import MeanIoU
from collections import namedtuple
import os

head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)
BatchEndParam = namedtuple('BatchEndParams', ['epoch', 'nbatch', 'eval_metric'])


def get_data():
    data_iter_to_predict = mx.io.ImageRecordIter_v1(
        # rec文件的路径
        path_imgrec=path_imgrec,
        # iterator生成的每一个实例的shape
        data_shape=(3, resize_size[0], resize_size[1]),
        mean_r=rgb_mean[0],
        mean_g=rgb_mean[1],
        mean_b=rgb_mean[2],
        data_name='data',
        label_name='softmax_label',
        # batch的大小
        batch_size=args.batch_size,
        # 是否随机从原图中切取出一个data_shape大小
        rand_crop=False,
        # 是否随机水平反射
        rand_mirror=False,
        label_width=resize_size[0] * resize_size[1],
        data_nthreads=50,
        shuffle=False,
    )

    # data_iter_to_save = mx.io.ImageRecordIter(
    #     # rec文件的路径
    #     path_imgrec=path_imgrec,
    #     # iterator生成的每一个实例的shape
    #     data_shape=(3, resize_size[0], resize_size[1]),
    #     data_name='data',
    #     label_name='softmax_label',
    #     # batch的大小
    #     batch_size=args.batch_size,
    #     # 是否随机从原图中切取出一个data_shape大小
    #     rand_crop=False,
    #     # 是否随机水平反射
    #     rand_mirror=False,
    #     label_width=resize_size[0] * resize_size[1],
    #     data_nthreads=50,
    #     shuffle=False,
    # )

    return data_iter_to_predict  # , data_iter_to_save


def get_mask(segs):
    for i in range(segs.shape[0]):
        for j in range(segs.shape[1]):
            if segs[i, j] == 1:
                mask_array[i, j, 3] = 200
            else:
                mask_array[i, j, 3] = 0
    mask = Image.fromarray(mask_array)
    return mask


def main():
    data_iter_to_predict = get_data()
    if args.test_io:
        tic = time.time()
        for i, batch in enumerate(data_iter_to_predict):
            for j in batch.data:
                j.wait_to_read()
            if (i + 1) % args.disp_batches == 0:
                logging.info('Batch [%d]\tSpeed: %.2f samples/sec' % (
                    i, args.disp_batches * args.batch_size / (time.time() - tic)))
                tic = time.time()

        return

    fcnxs, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(model_previx, epoch)
    for k, v in fcnxs_args.items():
        fcnxs_args[k] = v.copyto(ctx)
    # print fcnxs
    # quit()
    fcnxs_args["data"] = mx.nd.empty(data_iter_to_predict.provide_data[0][1], ctx)
    fcnxs_args["softmax_label"] = mx.nd.empty(data_iter_to_predict.provide_label[0][1], ctx)

    executor = fcnxs.bind(
        ctx,
        fcnxs_args,
        args_grad=None,
        grad_req="null",
        aux_states=fcnxs_auxs
    )
    # quit()
    logging.info(" in eval process...")
    eval_metric.reset()
    all_start = time.time()
    batch_end_callback = mx.callback.Speedometer(
        batch_size=args.batch_size,
        frequent=50,
        auto_reset=False)
    data_iter_to_predict.reset()
    for nbatch, data in enumerate(data_iter_to_predict):
        nbatch += 1
        # label_shape = data.label.shape
        # print data.index
        # quit()

        fcnxs_args["data"][:] = data.data[0]
        # fcnxs_args["softmax_label"][:] = data.label[0]

        executor.forward(is_train=False)
        pred_shape = executor.outputs[0].shape
        # print pred_shape
        # quit()

        cpu_output_array = mx.nd.empty(pred_shape)
        executor.outputs[0].copyto(cpu_output_array)

        if visual:
            # 预测
            fcnxs_segs = cpu_output_array.asnumpy()[:, 1] > args.threshold
            img_index = data.index
            for ind, (img_arr, seg) in enumerate(zip(data.data[0].asnumpy(),
                                                     fcnxs_segs)):
                mask = get_mask(seg)
                # print mask.size, img_arr.shape
                img = Image.fromarray((img_arr.transpose((1, 2, 0)) + rgb_mean).astype(np.uint8))
                img.paste(box=[0, 0], im=mask, mask=mask)
                img_name = idx2imgname[img_index[ind]]
                seg = '%s/%s' % (result_path, os.path.split(img_name)[-1])
                # print seg
                # quit()
                img.save(seg)
        # quit()
        label = data.label[0]
        pred = cpu_output_array.reshape((pred_shape[0],
                                         pred_shape[1],
                                         pred_shape[2] * pred_shape[3]))

        eval_metric.update([label], [pred])
        # print eval_metric.get()
        batch_end_params = BatchEndParam(epoch=epoch,
                                         nbatch=nbatch,
                                         # eval_metric=None,
                                         eval_metric=eval_metric,
                                         )
        batch_end_callback(batch_end_params)

        # if nbatch>200:
        #     quit()
        # quit()
        # self.executor.outputs[0].wait_to_read()
    name, value = eval_metric.get()
    logging.info('Validation-%s=%f', name, value)
    logging.info('eval time: %s s' % (time.time() - all_start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation FCN model.')

    parser.add_argument('--test-io', type=int, default=0,
                        help='1 means test reading speed without training')
    parser.add_argument('--disp-batches', type=int, default=20,
                        help='show progress for every n batches')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='the batch size')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='gpus to run, e.g. 0 or 1. -1 means using cpu')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='>threshold is True')
    parser.add_argument('--visual', type=bool, default=False,
                        help='save image to disk')
    args = parser.parse_args()
    logging.info(args)
    logging.info('start with arguments %s', args)

    if args.gpu == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)

    size_size = (256, 256)

    # model_previx = "FCN8s_VGG16"
    rgb_mean = (123.68, 116.779, 103.939)
    # batch_size = 20
    # portrait, coco, pascal, lip
    data_type = 'lip', 'val'

    # region 在这里设置模型
    lr = 1e-8
    pre_train_model_type = 'FCN8s'
    model_previx = 'model_coco_person/%s_VGG16_size%d_batch%d_lr%.0e' % (
        pre_train_model_type,
        size_size[0],
        8,
        lr
    )
    epoch = 75
    # lr = 1e-6
    # pre_train_model_type = 'FCN16s'
    # model_previx = 'model_coco_person/%s_VGG16_size%d_batch%d_lr%.0e' % (
    #     pre_train_model_type,
    #     resize_size[0],
    #     8,
    #     lr
    # )
    # epoch = 55
    # lr = 1e-6
    # model_previx = 'model_coco_person/FCN32s_VGG16_size%d_batch%d_lr%.0e' % (
    #     resize_size[0],
    #     8,
    #     lr
    # )
    # epoch = 50
    # lr = 1e-4
    # model_previx = 'model_pascal_person/FCN32s_VGG16_size%d_batch%d_lr%.0e' % (
    #     resize_size[0],
    #     8,
    #     lr
    # )
    # epoch = 200
    #
    # model_previx = 'model_pascal/FCN8s_VGG16'
    # epoch = 19

    logging.info('使用的模型：%s, epoch: %d' % (model_previx, epoch))
    # endregion
    # region 数据
    resize_size = (256, 256)
    path_imglst = "./segmentation_data/%s_person_datasets/%s_%s.lst" % (
        data_type[0],
        data_type[0], data_type[1])
    # path_imgrec = "./%s_person_datasets/%s_train_%d,%d.rec" % (data_type,data_type, resize_size[0], resize_size[1])
    path_imgrec = "./segmentation_data/%s_person_datasets/%s_%s_%d,%d.rec" % (
        data_type[0],
        data_type[0], data_type[1], resize_size[0], resize_size[1])
    logging.info('数据集：%s' % path_imgrec)
    idx2imgname = {}
    with open(path_imglst, 'r') as fin:
        idx2imgname = {int(line.split('\t')[0]): line.split('\t')[2].strip() for line in fin.readlines()}
    logging.info('图片数量：%d' % len(idx2imgname))
    # endregion

    eval_metric = 'meanIOU'
    if eval_metric == 'acc':
        eval_metric = metric.create(eval_metric)
    elif eval_metric == 'meanIOU':
        # eval_metric = MeanIoU(c=15, threshold=args.threshold, num_class=21)
        eval_metric = MeanIoU(c=1, threshold=args.threshold, num_class=2)

    visual = args.visual

    if visual:
        result_path = './result/%s_person_datasets/result_%s_epoch%d_%s' % (
            data_type[0],
            pre_train_model_type, epoch, data_type[1])
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        logging.info('结果保存到：%s' % result_path)
        # 产生一个颜色和mask
        rgb = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        # print rgb
        for i in range(3):
            rgb[i] = int(rgb[i] * 255)
        rgb = tuple(rgb)
        # print rgb
        # 产生一个mask
        mask = Image.new("RGB", color=rgb, size=(resize_size[1], resize_size[0]))
        mask = mask.convert("RGBA")
        mask_array = np.array(mask)
    main()
