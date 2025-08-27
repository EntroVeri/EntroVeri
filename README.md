# EntroVeri

## Introduction
This is the official implementation for our paper "EntroVeri: A Harmless and Efficient Black-box Watermarking Method for Model Ownership Verification". This project is developed on Python3 and Pytorch.

## Getting Start

### Settings
First, create a virtual environment using Anaconda.
<pre lang="markdown">conda create -n EntroVeri python=3.8.10
conda activate EntroVeri</pre>

Second, install the necessary packages to run EntroVeri.
<pre lang="markdown">conda install pytorch==2.0.0 torchvision==0.15.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install numpy==1.24.2
pip install scipy==1.10.1
pip install Pillow==9.4.0
pip install tqdm==4.61.2
</pre>

Third, run the following command twice to get 2 different benign models, one used for thresholding and one used for evaluation/varification.
<pre lang="markdown">python benign_model.py
</pre>

### Watermark Embedding
Run the following command to get a watermarked model.
<pre lang="markdown">python 1-embedding.py
</pre>

### Entropy Thresholding
Run the following command to get the entropy threshold.
<pre lang="markdown">python 2-thresholding.py
</pre>

### Verification
Run the following command to verify the ownership across 5 cases.
<pre lang="markdown">python 3-verifiation.py
</pre>

