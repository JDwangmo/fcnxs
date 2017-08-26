#!/usr/bin/env bash

while getopts h:g: option
do
    case "$option" in
        h)
            echo "option:h, value $OPTARG"
            echo "next arg index:$OPTIND"
            exit 1;;
        g)
            echo "gpu:$OPTARG"
            GPU=$OPTARG
            ;;
        s)
            echo "option:s"
            echo "next arg index:$OPTIND"
            exit 1;;
        \?)
            echo "Usage: args [-h n] [-m] [-s]"
            echo "-h means hours"
            echo "-m means minutes"
            echo "-s means seconds"
            exit 1;;
    esac
done


# train fcn-32s model
#python -u fcn_xs_person.py --model=FCN32s --prefix=VGG_FC_ILSVRC_16_layers \
#       --epoch=74 --init-type=vgg16 --gpu=${GPU}
python -u fcn_xs_person.py --model=FCN32s --init-type=fcnxs --gpu=${GPU}
# train fcn-16s model
#python -u fcn_xs_person.py --model=FCN16s --epoch=30 --init-type=fcnxs

# train fcn-8s model
#python -u fcn_xs_person.py --model=FCN8s --epoch=100 --init-type=fcnxs --gpu=2

# train FCN_atrous model
#python -u fcn_xs_person.py --model=FCN_atrous --epoch=80 --init-type=fcnxs --gpu=3

# train fcn-4s model
#python -u fcn_xs_person.py --model=FCN4s --epoch=100 --init-type=fcnxs --gpu=${GPU}

#
#python -u fcn_xs.py --model=fcn8s --prefix=FCN16s_VG \
#      --epoch=27 --init-type=fcnxs
