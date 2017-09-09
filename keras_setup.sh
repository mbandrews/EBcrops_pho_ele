#!/bin/bash

# Install Miniconda environment manager
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

########################################
 
# Create keras-enabled conda environment
#envname=hep
envname=hep-gpu
conda env create -f ${envname}.yml
#conda remove --name $envname --all

# Verify Tensorflow installation
source activate $envname
python -c "import tensorflow as tf; tf.Session(config=tf.ConfigProto(log_device_placement=True))"
#python -c "import tensorflow as tf; hello = tf.constant('Hello, TensorFlow!'); sess = tf.Session(); print(sess.run(hello))"
source deactivate $envname
