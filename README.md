# Code for Aberration Correction for Mass-Produced Mobile Cameras in MIPI Challenge
This repository contains both training code and evalutation code for [Aberration Correction for Mass-Produced Mobile Cameras in MIPI Challenge](https://www.codabench.org/competitions/8417/). The Aberration Correction for Mass-Produced Mobile Cameras Challenge is one track of MIPI-Challenge, Mobile Intelligent Photography & Imaging Workshop 2025, in conjunction with ICCV 2025. Participants are restricted to train their algorithms on the provided dataset. Participants are expected to develop more robust and generalized methods for aberration correction for mass-produced mobile cameras.


## Dataset Preparation
- Download the **AberrationCorrection-MIPI-dataset** ([Baidu Disk(vgu2)](https://pan.baidu.com/s/1I8VwHBA51Z726WVf43lISA?pwd=vgu2)/[Google Drive](https://drive.google.com/drive/folders/1vxyp5uDU7OId6_5laGYi-JHJsB3E1jmz?usp=sharing)) and unzip the images in datasets/train and datasets/valid folders.
- Arrange the folders as follows: 
  
```
|-- datasets
  |-- train
    |-- train_dataset_pertubed.h5
  |-- valid
    |-- lq
    |-- wb_params.npz
```
### Training Data Loading

The method for loading training data can be referred to in the `fewlens/data/h5_dataset.py` file. For each training sample, the `item` contains multiple channels. The data is split as follows:

```python
lq = item[:3]        # The first three channels represent the Low-Quality (LQ) image
gt = item[3:6]        # Channels 4 to 6 represent the Ground-Truth (GT) image
fld_info = item[6:8]  # Channels 7 and 8 represent normalized (x, y) coordinate information
```
``fld_info`` **Explanation**
The fld_info field encodes the spatial position of each pixel using its (x, y) coordinates, normalized to the range ``[-1, 1]``.

**Normalization Formula**
For an image of height ``h`` and width ``w``, the normalized coordinates are computed as:

```
x_normalized = (x - h / 2) / h
y_normalized = (y - w / 2) / w
```
Definition of Terms
``x``, ``y``: Original pixel coordinates

``h``, ``w``: Height and width of the image

``x_normalized``, ``y_normalized``: Normalized spatial coordinates in the range ``[-1, 1]``

## Dependencies and Installation
```bash
conda create -n fewlens python=3.9
conda activate fewlens
pip install -r requirements.txt
cd fewlens/dcn
BASICSR_EXT=True python setup.py develop
```

## Train

```bash
sh train_mimo_pertubed.sh
```
## Inference
```
export PYTHONPATH="./:${PYTHONPATH}" 
python inference/inference_mimo.py -w model_path -i datasets/valid/lq
```

## Citation

If you find this repository useful in your project, please consider giving a :star:. We will release the paper for citation as soon as possible.

## Contact

For technical questions, please contact `fujiayi[AT]mail.nankai.edu.cn`
