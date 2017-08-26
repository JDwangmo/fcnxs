# coding: utf-8
from __future__ import absolute_import
from mxnet.metric import EvalMetric, check_label_shapes
import math
from collections import OrderedDict
import numpy
from mxnet import ndarray
import mxnet as mx
import time


class PrecisionRecall(EvalMetric):
    """Computes precision and recall rate for Specified class.

    TP: number of groun truth=c and prediction=c
    FN: number of groun truth=c and prediction!=c
    FP: number of groun truth!=c and prediction=c
    TN: number of groun truth!=c and prediction!=c

    precision = TP / (TP + FP)
    recall rate = TP / (TP + FN)

    Parameters
    ----------
    c : int
        specified class to compute precision and recall rate.
    axis : axis : int, default=1
        The axis that represents classes.
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """

    def __init__(self, c, axis=1, name='precision and recall rate',
                 name_1='precision', name_2='recall rate',
                 output_names=None, label_names=None):
        super(PrecisionRecall, self).__init__(
            name, axis=axis,
            output_names=output_names, label_names=label_names)
        self.name_1 = name_1
        self.name_2 = name_2
        self.axis = axis
        self.c = c

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        check_label_shapes(labels, preds)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = ndarray.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')

            check_label_shapes(label, pred_label)

            pred_label_ = pred_label.flat
            label_ = label.flat
            for i in range(len(pred_label_)):
                if label_[i] == self.c:
                    if pred_label_[i] == self.c:
                        self.TP += 1
                    else:
                        self.FN += 1
                elif pred_label_[i] == self.c:
                    self.FP += 1

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.TP = 0.0
        self.FN = 0
        self.FP = 0

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        return ([self.name_1, self.name_2],
                [self.TP / (self.TP + self.FP), self.TP / (self.TP + self.FN)])


class MeanIoU(EvalMetric):
    """Mean Intersection over Union.
    Computes a ratio between intersection of ground truth and predicted class over
    union of ground truth and predicted class.

    Parameters
    ----------
    c : int or None
        compute the Mean IoU of specified class.
        if c == None, compute the average mean IoU of all classes.
    axis : axis : int, default=1
        The axis that represents classes.
    num_class : int, default=2
        number of classes.
        it will be used when computing average mean IoU of all classes.
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all lebels.
    """

    def __init__(self, c=1, axis=1, num_class=2, threshold=0.5, name='mean_IoU',
                 output_names=None, label_names=None):
        super(MeanIoU, self).__init__(
            name, axis=axis,
            output_names=output_names, label_names=label_names)
        self.c = c
        if self.c is not None:
            self.name = 'mean IoU of class ' + str(self.c)
        self.axis = axis
        self.num_class = num_class
        self.threshold = threshold
        self.TP = 0.0
        self.FN = 0
        self.FP = 0

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            # start_all = time.time()
            pred = pred.reshape((pred.shape[0], pred.shape[1], -1))
            # print label.shape, pred.shape
            if pred.shape != label.shape:
                # start = time.time()
                if self.threshold == 0.5 and self.num_class <= 2:
                    pred = ndarray.argmax(pred, axis=self.axis)
                else:
                    pred = pred[:, self.c, :] > self.threshold
                    # end = time.time()
                    # print 'bb2: %f' % (end - start)
            # print pred
            # check_label_shapes(label, pred)
            if label.shape != pred.shape:
                raise ValueError("Shape of labels {} does not match shape of "
                                 "predictions {}".format(label.shape, pred.shape))
            # print label.shape
            # print pred_label.shape
            # end = time.time()
            # print 'bb: %f' % (end - start)
            if self.c is not None:
                pred = pred.copyto(mx.cpu(0))
                # print pred.context
                # print label.context
                pred = ndarray.flatten(pred)
                label = ndarray.flatten(label)
                # print pred.shape, label.shape
                p_1 = pred == 1
                p_0 = pred == 0
                l_1 = label == 1
                l_0 = label == 0
                # quit()
                TP = ndarray.sum((p_1 * l_1), axis=[0, 1])
                FP = ndarray.sum(p_1 * l_0, axis=[0, 1])
                FN = ndarray.sum(p_0 * l_1, axis=[0, 1])
                # start = time.time()
                self.TP += TP
                self.FP += FP
                self.FN += FN
                # print TP,FP,FN
                # print self.TP, self.FP, self.FN
                # self.TP += TP.asnumpy()[0]
                # self.FP += FP.asnumpy()[0]
                # self.FN += FN.asnumpy()[0]
                # end = time.time()
                # print 'cc2: %f' % (end - start)
                # quit()
                # print 'cc1: %f' % (end - start)
                # region 效率太低
                # start = time.time()
                # pred = pred.asnumpy().astype('int32')
                # label = label.asnumpy().astype('int32')
                # pred_label_ = pred.flat
                # label_ = label.flat
                # for i in range(len(pred_label_)):
                #     if label_[i] == self.c:
                #         if pred_label_[i] == self.c:
                #             self.TP += 1
                #         else:
                #             self.FN += 1
                #     elif pred_label_[i] == self.c:
                #         self.FP += 1
                # print self.TP, self.FP, self.FN
                # self.MeanIoU.append(float(self.TP) / (self.TP + self.FN + self.FP))
                # endregion
            else:
                # todo：这部分代码有问题
                pass
                # #self.MeanIoU = [0.0]*self.num_class
                # for class_num in range(self.num_class):
                #     for i in range(len(pred_label_)):
                #         if label_[i] == class_num:
                #             if pred_label_[i] == class_num:
                #                 self.TP += 1
                #             else:
                #                 self.FN += 1
                #         elif pred_label_[i] == class_num:
                #             self.FP += 1
                #     #self.MeanIoU.append(float(self.TP) / (self.TP + self.FN + self.FP))
                #     #self.MeanIoU[class_num] = float(self.TP) / (self.TP + self.FN + self.FP)
                #     self.TP = self.FN = self.FP = 0.0
                # print self.TP, self.FN, self.FP
                # quit()
                # end_all = time.time()
                # print end_all - start_all
                # quit()

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.TP = 0.0
        self.FN = 0
        self.FP = 0
        # self.MeanIoU = []

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        # return self.name, sum(self.MeanIoU) / len(self.MeanIoU)
        # print self.TP/(self.TP + self.FN + self.FP)
        # print(self.name, (self.TP.asnumpy(), (self.TP + self.FN + self.FP).asnumpy()))
        return self.name, (self.TP / (self.TP + self.FN + self.FP)).asnumpy()
