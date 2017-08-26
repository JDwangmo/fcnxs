#!/usr/bin/env bash

make_list=0
#data_type='test'
#data_type='coco'
#data_type='portrait'
data_type='pascal'
num_thread=20
#
if [ ${make_list} -eq 1 ]
then
    # make list
    if [ ${data_type} == 'pascal' ]
    then
        # train and val
        python im2rec.py --list=True pascal_train ~/Projects/fcn-xs/pascal_voc_person_datasets/images/ pascal \
            --num-thread=10
        python im2rec.py --list=True pascal_val ~/Projects/fcn-xs/pascal_voc_person_datasets/images/ pascal \
            --num-thread=10
    elif [ ${data_type} = 'portrait' ]
    then
        python im2rec.py --list=True potrait_person_datasets/portrait \
            ~/Projects/fcn-xs/potrait_person_datasets/images/ portrait \
            --num-thread=10 --train-ratio=0.7
    else
        # train
        python im2rec.py --list=True coco_train ~/Projects/fcn-xs/coco_person_datasets/images_train/ coco \
            --num-thread=10
        # test
        python im2rec.py --list=True coco_val ~/Projects/fcn-xs/coco_person_datasets/images_val/ coco \
            --num-thread=10

    fi
else
    SIZE='800,600'
#    SIZE='128,128'
    echo ${SIZE}

    if [ ${data_type} = 'pascal' ]
    then
        # train
        python im2rec.py pascal_voc_person_datasets/pascal_train pascal_voc_person_datasets/ pascal \
            --num-thread=20 --pack-label=1 --resize_to=${SIZE} --quality=100

        mv pascal_voc_person_datasets/pascal_train.rec pascal_voc_person_datasets/pascal_train_${SIZE}.rec
        mv pascal_voc_person_datasets/pascal_train.idx pascal_voc_person_datasets/pascal_train_${SIZE}.idx
#
#        # val
        python im2rec.py pascal_voc_person_datasets/pascal_val pascal_voc_person_datasets/ pascal \
            --num-thread=20 --pack-label=1 --resize_to=${SIZE} --quality=100
#
        mv pascal_voc_person_datasets/pascal_val.rec pascal_voc_person_datasets/pascal_val_${SIZE}.rec
        mv pascal_voc_person_datasets/pascal_val.idx pascal_voc_person_datasets/pascal_val_${SIZE}.idx
    elif [ ${data_type} = 'portrait' ]
    then
        # train
        python im2rec.py portrait_person_datasets/portrait_train portrait_person_datasets/images/ portrait \
            --num-thread=${num_thread} --pack-label=1 --resize_to=${SIZE} --quality=100

        mv portrait_person_datasets/portrait_train.rec portrait_person_datasets/portrait_train_${SIZE}.rec
        mv portrait_person_datasets/portrait_train.idx portrait_person_datasets/portrait_train_${SIZE}.idx
#         val
        python im2rec.py portrait_person_datasets/portrait_val portrait_person_datasets/images/ portrait \
            --num-thread=${num_thread} --pack-label=1 --resize_to=${SIZE} --quality=100

        mv portrait_person_datasets/portrait_val.rec portrait_person_datasets/portrait_val_${SIZE}.rec
        mv portrait_person_datasets/portrait_val.idx portrait_person_datasets/portrait_val_${SIZE}.idx
    else
        # train
        python im2rec.py coco_person_datasets/coco_train ~/Projects/fcn-xs/coco_person_datasets/images_train/ coco \
            --num-thread=20 --pack-label=True --resize_to=${SIZE} --quality=100

        mv coco_person_datasets/coco_train.rec coco_person_datasets/coco_train_${SIZE}.rec
        mv coco_person_datasets/coco_train.idx coco_person_datasets/coco_train_${SIZE}.idx

        # val
        python im2rec.py coco_person_datasets/coco_val ~/Projects/fcn-xs/coco_person_datasets/images_val/ coco \
            --num-thread=20 --pack-label=True --resize_to=${SIZE} --quality=100

        mv coco_person_datasets/coco_val.rec coco_person_datasets/coco_val_${SIZE}.rec
        mv coco_person_datasets/coco_val.idx coco_person_datasets/coco_val_${SIZE}.idx
    fi
fi