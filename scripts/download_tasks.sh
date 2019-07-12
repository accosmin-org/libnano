#!/bin/bash

dir_exp=$HOME/experiments/results
dir_data=$HOME/experiments/datasets

dir_data_iris=${dir_data}/iris/
dir_data_wine=${dir_data}/wine/
dir_data_mnist=${dir_data}/mnist/
dir_data_cifar10=${dir_data}/cifar10/
dir_data_cifar100=${dir_data}/cifar100/
dir_data_fashion_mnist=${dir_data}/fashion-mnist/

# MNIST dataset
function download_mnist {
	local dir=${dir_data}/mnist
	mkdir -p ${dir}

	wget -N http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P ${dir}
	wget -N http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P ${dir}
	wget -N http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P ${dir}
	wget -N http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P ${dir}
}

# Fashion-MNIST dataset
function download_fashion_mnist {
	local dir=${dir_data}/fashion-mnist
	mkdir -p ${dir}

	wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz -P ${dir}
	wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz -P ${dir}
	wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz -P ${dir}
	wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz -P ${dir}
}

# CIFAR10 dataset
function download_cifar10 {
	local dir=${dir_data}/cifar10
	mkdir -p ${dir}

	wget -N http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -P ${dir}
}

# CIFAR100 dataset
function download_cifar100 {
	local dir=${dir_data}/cifar100
	mkdir -p ${dir}

	wget -N http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz -P ${dir}
}

# IRIS dataset
function download_iris {
	local dir=${dir_data}/iris
	mkdir -p ${dir}

	wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data -P ${dir}
}

# WINE dataset
function download_wine {
	local dir=${dir_data}/wine
	mkdir -p ${dir}

	wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data -P ${dir}
}

# IRIS dataset
function download_iris {
	local dir=${dir_data}/iris
	mkdir -p ${dir}

	wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data -P ${dir}
	wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names -P ${dir}
}

# CALIFORNIA housing dataset
function download_cal_housing {
    local dir=${dir_data}/cal-housing
    mkdir -p ${dir}

    wget -N http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz -P ${dir}
    tar xvf ${dir}/cal_housing.tgz -C ${dir}
    mv ${dir}/CaliforniaHousing/* ${dir}/
}

# Process command line
function usage {
	cat <<EOF
usage: $0 [OPTIONS]

options:
    -h,--help
        print usage
    --mnist
        download MNIST dataset
    --fashion-minst
        download Fashion-MNIST dataset
    --iris
        download IRIS dataset
    --wine
        download WINE dataset
    --cifar10
        download CIFAR-10 dataset
    --cifar100
        download CIFAR-100 dataset
    --cal-housing
        download California Housing dataset
EOF
	exit 1
}

if [ "$1" == "" ]; then
	usage
fi

while [ "$1" != "" ]; do
	case $1 in
		-h | --help)	    usage
					        ;;
		--wine)			    download_wine
					        ;;
		--iris)			    download_iris
					        ;;
		--mnist)		    download_mnist
					        ;;
		--fashion-mnist)	download_fashion_mnist
					        ;;
		--cifar10)		    download_cifar10
					        ;;
		--cifar100)		    download_cifar100
					        ;;
        --cal-housing)      download_cal_housing
                            ;;
		*)			        echo "unrecognized option $1"
					        echo
				        	usage
					        ;;
	esac
	shift
done

exit 0
