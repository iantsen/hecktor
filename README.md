# 1st Place Solution for the [HECKTOR](https://www.aicrowd.com/challenges/miccai-2020-hecktor) challenge

> The official implementation of the winning solution for the MICCAI 2020 HEad and neCK TumOR segmentation challenge (HECKTOR).

### Main requirements
- PyTorch 1.6.0 (cuda 10.2)
- SimpleITK 1.2.4 (ITK 4.13)
- nibabel 3.1.1
- skimage 0.17.2

### Dataset
Train and test images are available through the competition [website](https://www.aicrowd.com/challenges/miccai-2020-hecktor). The concise description of the dataset is present in `notebooks/make_dataset.ipynb`.     

### Data preprocessing
The data preprocessing consists of:
- Resampling the pair of PET & CT images for each patient to a common reference space.
- Extracting the region of interest (bounding box) of the size of 144x144x144 voxels. 
- Saving the transformed images in NIfTI format.

To prepare the dataset in _an interactive manner_, one can use `notebooks/make_dataset.ipynb`, that gives an explanation about each step.
Alternatively, _the fully automated data preprocessing_ can be performed by running `src/data/make_dataset.py`. All required parameters must be provided as _a single config file_ in the YAML data format: 
```sh
cd hecktor/src/data/
python make_dataset.py -p hecktor/config/make_dataset.yaml
```
Use `/config/make_dataset.yaml` to specify all required parameters.

### Training
For training the model from scratch, one can use `notebooks/model_train.ipynb` setting all parameters right in the notebook. Otherwise, with all parameters written in the config file, one needs to run `hecktor/model/train.py` from its current directory:
```sh
cd hecktor/model/
python train.py -p hecktor/config/model_train.yaml
```
All parameters are described in `hecktor/config/model_train.yaml` that should be used as a template to build your own config file.

### Inference
For inference, run the script `hecktor/model/predict.py` with parameters defined in the config file `hecktor/config/model_predict.yaml`:
```sh
cd hecktor/model/
python predict.py -p hecktor/config/model_predict.yaml
```

### Model weights
To reproduce results presented in the paper on different train / validation folds, one must download and save pretrained weights in the folder `hecktor/model/weights/`. Weights of a single model are stored in files named `{split}_best_model_weights.pt`. IDs of the patients in the train / validation folds for each data split are stored in `train_val_{split}.pkl` files located in the folder `hecktor/src/data/splits/`.

In order to download weights of all pretrained model (eight models in total) built on the different train / validation, use the following [link](https://www.dropbox.com/sh/kkvqwn0bpnt1ynk/AABGNdpzTSiIKjiGV5K2ta0Na?dl=0).


### Example
![img](https://drive.google.com/uc?export=view&id=1U5ifCqqWMKV65wvv1x2BWAKMvMZS9Ywt)


### Paper
If you use this code in you research, please cite the following paper ([arXiv](https://arxiv.org/abs/2102.10446)):
> Iantsen A., Visvikis D., Hatt M. (2021) Squeeze-and-Excitation Normalization for Automated Delineation of Head and Neck Primary Tumors in Combined PET and CT Images. In: Andrearczyk V., Oreiller V., Depeursinge A. (eds) Head and Neck Tumor Segmentation. HECKTOR 2020. Lecture Notes in Computer Science, vol 12603. Springer, Cham. https://doi.org/10.1007/978-3-030-67194-5_4
