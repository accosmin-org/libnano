#!/bin/bash

dir_exp=$HOME/experiments/results
dir_data=$HOME/experiments/datasets

function download_mnist {
    local dir=${dir_data}/mnist/
    mkdir -p ${dir}

    wget -N http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P ${dir}
    wget -N http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P ${dir}
    wget -N http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P ${dir}
    wget -N http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P ${dir}
}

function download_fashion_mnist {
    local dir=${dir_data}/fashion-mnist/
    mkdir -p ${dir}

    wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz -P ${dir}
    wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz -P ${dir}
    wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz -P ${dir}
    wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz -P ${dir}
}

function download_cifar10 {
    local dir=${dir_data}/cifar10/
    mkdir -p ${dir}

    wget -N http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -P ${dir}
}

function download_cifar100 {
    local dir=${dir_data}/cifar100/
    mkdir -p ${dir}

    wget -N http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz -P ${dir}
}

function download_iris {
    local dir=${dir_data}/iris/
    mkdir -p ${dir}

    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data -P ${dir}
    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names -P ${dir}
}

function download_wine {
    local dir=${dir_data}/wine/
    mkdir -p ${dir}

    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data -P ${dir}
    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.names -P ${dir}
}

function download_adult {
    local dir=${dir_data}/adult
    mkdir -p ${dir}

    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -P ${dir}
    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test -P ${dir}
    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names -P ${dir}
}

function download_forest_fires {
    local dir=${dir_data}/forest-fires
    mkdir -p ${dir}

    wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv -P ${dir}
    wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.names -P ${dir}
}

function download_poker_hand {
    local dir=${dir_data}/poker-hand
    mkdir -p ${dir}

    wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data -P ${dir}
    wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data -P ${dir}
    wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand.names -P ${dir}
}

function download_breast_cancer {
    local dir=${dir_data}/breast-cancer
    mkdir -p ${dir}

    wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data -P ${dir}
    wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names -P ${dir}
}

function download_abalone {
    local dir=${dir_data}/abalone
    mkdir -p ${dir}

    wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data -P ${dir}
    wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names -P ${dir}
}

function download_bank_marketing {
    local dir=${dir_data}/bank-marketing
    mkdir -p ${dir}

    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip -P ${dir}
    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip -P ${dir}

    unzip -o ${dir}/bank.zip -d ${dir}
    unzip -o ${dir}/bank-additional.zip -d ${dir}

    mv -f ${dir}/bank-additional/* ${dir}
    rm -rf ${dir}/__MACOSX
    rm -rf ${dir}/bank-additional
    rm -f ${dir}/*.zip
}

function download_all {
    download_mnist
    download_cifar10
    download_cifar100
    download_fashion_mnist

    download_iris
    download_wine
    download_adult
    download_abalone
    download_poker_hand
    download_forest_fires
    download_breast_cancer
    download_bank_marketing
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
    --poker-hand
        download Poker Hand dataset
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
        --all)              download_all
                            ;;
        --wine)             download_wine
                            ;;
        --iris)             download_iris
                            ;;
        --adult)            download_adult
                            ;;
        --mnist)            download_mnist
                            ;;
        --abalone)          download_abalone
                            ;;
        --fashion-mnist)    download_fashion_mnist
                            ;;
        --cifar10)          download_cifar10
                            ;;
        --cifar100)         download_cifar100
                            ;;
        --poker-hand)       download_poker_hand
                            ;;
        --forest-fires)     download_forest_fires
                            ;;
        --breast-cancer)    download_breast_cancer
                            ;;
        --bank-marketing)   download_bank_marketing
                            ;;
        *)                  echo "unrecognized option $1"
                            echo
                            usage
                            ;;
    esac
    shift
done

exit 0
