#!/usr/bin/env bash

dir_data=$HOME/libnano/datasets

function unzip_dir {
    local dir=$1

    for file in ${dir}/*.gz; do
        gunzip -fk ${file} || return 1
    done
}

function untar_dir {
    local dir=$1

    for file in ${dir}/*.tar.gz; do
        tar -xvf ${file} -C ${dir} || return 1
    done
}

function download_mnist {
    local dir=${dir_data}/mnist/
    mkdir -p ${dir}

    files=(
        "train-images.idx3-ubyte.gz"
        "train-labels.idx1-ubyte.gz"
        "t10k-images.idx3-ubyte.gz"
        "t10k-labels.idx1-ubyte.gz"
    )

    for file in "${files[@]}"; do
        wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/mnist-mld/${file} -P ${dir} || return 1
    done

    for file in `ls ${dir}/*.gz`; do
        mv ${file} ${file/.idx/-idx}
    done

    unzip_dir ${dir} || return 1
}

function download_fashion_mnist {
    local dir=${dir_data}/fashion-mnist/
    mkdir -p ${dir}

    wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz -P ${dir} || return 1
    wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz -P ${dir} || return 1
    wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz -P ${dir} || return 1
    wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz -P ${dir} || return 1

    unzip_dir ${dir} || return 1
}

function download_cifar10 {
    local dir=${dir_data}/cifar10/
    mkdir -p ${dir}

    wget -N http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -P ${dir} || return 1

    untar_dir ${dir} || return 1
}

function download_cifar100 {
    local dir=${dir_data}/cifar100/
    mkdir -p ${dir}

    wget -N http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz -P ${dir} || return 1

    untar_dir ${dir} || return 1
}

function download_iris {
    local dir=${dir_data}/iris/
    mkdir -p ${dir}

    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data -P ${dir} || return 1
    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names -P ${dir} || return 1
}

function download_wine {
    local dir=${dir_data}/wine/
    mkdir -p ${dir}

    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data -P ${dir} || return 1
    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.names -P ${dir} || return 1
}

function download_adult {
    local dir=${dir_data}/adult
    mkdir -p ${dir}

    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -P ${dir} || return 1
    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test -P ${dir} || return 1
    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names -P ${dir} || return 1
}

function download_forest_fires {
    local dir=${dir_data}/forest-fires
    mkdir -p ${dir}

    wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv -P ${dir} || return 1
    wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.names -P ${dir} || return 1
}

function download_breast_cancer {
    local dir=${dir_data}/breast-cancer
    mkdir -p ${dir}

    wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data -P ${dir} || return 1
    wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names -P ${dir} || return 1
}

function download_abalone {
    local dir=${dir_data}/abalone
    mkdir -p ${dir}

    wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data -P ${dir} || return 1
    wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names -P ${dir} || return 1
}

function download_bank_marketing {
    local dir=${dir_data}/bank-marketing
    mkdir -p ${dir}

    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip -P ${dir} || return 1
    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip -P ${dir} || return 1

    unzip -o ${dir}/bank.zip -d ${dir} || return 1
    unzip -o ${dir}/bank-additional.zip -d ${dir} || return 1

    mv -f ${dir}/bank-additional/* ${dir}
    rm -rf ${dir}/__MACOSX
    rm -rf ${dir}/bank-additional
    rm -f ${dir}/*.zip
}

function download_all {
    download_mnist || return 1
    download_cifar10 || return 1
    download_cifar100 || return 1
    download_fashion_mnist || return 1

    download_iris || return 1
    download_wine || return 1
    download_adult || return 1
    download_abalone || return 1
    download_forest_fires || return 1
    download_breast_cancer || return 1
    download_bank_marketing || return 1
}

# Process command line
function usage {
    cat <<EOF
usage: $0 [OPTIONS]

options:
    -h,--help
        print usage
    --all
        download all datasets
    --mnist
        download MNIST dataset
    --fashion-minst
        download Fashion-MNIST dataset
    --iris
        download Iris dataset
    --wine
        download Wine dataset
    --adult
        download Adult dataset
    --abalone
        download Abalone dataset
    --cifar10
        download CIFAR-10 dataset
    --cifar100
        download CIFAR-100 dataset
    --forest-fires
        download Forest Fires dataset
    --breast-cancer
        download Breast Cancer dataset
    --bank-marketing
        download Bank Marketing dataset
EOF
    exit 1
}

if [ "$1" == "" ]; then
    usage
fi

while [ "$1" != "" ]; do
    case $1 in
        -h | --help)        usage
                            ;;
        --all)              download_all || exit 1
                            ;;
        --wine)             download_wine || exit 1
                            ;;
        --iris)             download_iris || exit 1
                            ;;
        --adult)            download_adult || exit 1
                            ;;
        --mnist)            download_mnist || exit 1
                            ;;
        --abalone)          download_abalone || exit 1
                            ;;
        --fashion-mnist)    download_fashion_mnist || exit 1
                            ;;
        --cifar10)          download_cifar10 || exit 1
                            ;;
        --cifar100)         download_cifar100 || exit 1
                            ;;
        --poker-hand)       download_poker_hand || exit 1
                            ;;
        --forest-fires)     download_forest_fires || exit 1
                            ;;
        --breast-cancer)    download_breast_cancer || exit 1
                            ;;
        --bank-marketing)   download_bank_marketing || exit 1
                            ;;
        *)                  echo "unrecognized option $1"
                            echo
                            usage
                            ;;
    esac
    shift
done

exit 0
