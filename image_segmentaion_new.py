# pylint: skip-file
import numpy as np
import mxnet as mx
from PIL import Image
from matplotlib import pyplot as plt
import os


# from PythonAPI.pycocotools.coco import COCO

def getpallete(num_cls):
    # this function is to get the colormap for visualizing the segmentation mask
    n = num_cls
    pallete = [0] * (n * 3)
    for j in xrange(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while (lab > 0):
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete


pallete = getpallete(256)
img_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
img_prefix = "./images_data_crop/"
model_previx = "FCN16s_VGG16"
epoch = 24
ctx = mx.gpu(1)


def get_data(img_path):
    """get the (1, 3, h, w) np.array data for the img_path"""
    mean = np.array([123.68, 116.779, 103.939])  # (R,G,B)
    img = Image.open(img_path)
    img = np.array(img, dtype=np.float32)
    reshaped_mean = mean.reshape(1, 1, 3)
    img = img - reshaped_mean
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = np.expand_dims(img, axis=0)
    return img


def main():
    fcnxs, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(model_previx, epoch)
    for k, v in fcnxs_args.items():
        fcnxs_args[k] = v.copyto(ctx)
    exector = fcnxs.bind(ctx, fcnxs_args, args_grad=None, grad_req="null", aux_states=fcnxs_auxs)
    for i in range(1, 100):
        if i % 10 == 0:
            print i
        img_path = img_prefix + '0' * (5 - len(str(i))) + str(i) + ".jpg"
        if os.path.exists(img_path) == False:
            continue
        img_data = get_data(img_path)
        fcnxs_args["data"] = mx.nd.array(img_data, ctx)
        data_shape = fcnxs_args["data"].shape
        label_shape = (1, data_shape[2] * data_shape[3])
        fcnxs_args["softmax_label"] = mx.nd.empty(label_shape, ctx)
        # exector = fcnxs.bind(ctx, fcnxs_args ,args_grad=None, grad_req="null", aux_states=fcnxs_auxs)
        exector.forward(is_train=False)
        output = exector.outputs[0][0, 1]

        rgb = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        for i in range(3):
            rgb[i] = int(rgb[i] * 255)
        rgb = tuple(rgb)

        img = Image.open(img_path)
        output = output.asnumpy()

        mask = Image.new("RGB", color=rgb, size=img.size)
        mask = mask.convert("RGBA")
        mask_array = np.array(mask)

        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if output[i, j] > 0.5:
                    mask_array[i, j, 3] = 200
                else:
                    mask_array[i, j, 3] = 0

        mask = Image.fromarray(mask_array)
        img.paste(box=[0, 0], im=mask, mask=mask)

        # out_img = np.uint8(np.squeeze(output.asnumpy().argmax(axis=1)))
        # out_img = Image.fromarray(out_img)
        # out_img.putpalette(pallete)
        seg = img_path.replace('.jpg', '.png').replace('./images_data_crop/', '')
        seg = './segment_img/' + seg
        img.save(seg)


if __name__ == "__main__":
    main()
