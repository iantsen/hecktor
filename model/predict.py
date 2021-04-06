import sys
import argparse
import yaml
import pathlib

import torch
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True

sys.path.append('../src/')
sys.path.append('../src/data/')
import dataset
import transforms
import utils
import models
import predictor


def main(args):
    path_to_config = pathlib.Path(args.path)
    with open(path_to_config) as f:
        config = yaml.safe_load(f)

    # read config:
    path_to_data = pathlib.Path(config['path_to_data'])
    path_to_save_dir = pathlib.Path(config['path_to_save_dir'])
    path_to_weights = config['path_to_weights']
    probs = config['probs']
    num_workers = int(config['num_workers'])
    n_cls = int(config['n_cls'])
    in_channels = int(config['in_channels'])
    n_filters = int(config['n_filters'])
    reduction = int(config['reduction'])

    # test data paths:
    all_paths = utils.get_paths_to_patient_files(path_to_imgs=path_to_data, append_mask=False)

    # input transforms:
    input_transforms = transforms.Compose([
        transforms.NormalizeIntensity(),
        transforms.ToTensor(mode='test')
    ])

    # ensemble output transforms:
    output_transforms = [
        transforms.InverseToTensor(),
        transforms.CheckOutputShape(shape=(144, 144, 144))
    ]
    if not probs:
        output_transforms.append(transforms.ProbsToLabels())

    output_transforms = transforms.Compose(output_transforms)

    # dataset and dataloader:
    data_set = dataset.HecktorDataset(all_paths, transforms=input_transforms, mode='test')
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=num_workers)

    # model:
    model = models.FastSmoothSENormDeepUNet_supervision_skip_no_drop(in_channels, n_cls, n_filters, reduction)

    # init predictor:
    predictor_ = predictor.Predictor(
        model=model,
        path_to_model_weights=path_to_weights,
        dataloader=data_loader,
        output_transforms=output_transforms,
        path_to_save_dir=path_to_save_dir
    )

    # check if multiple paths were provided to run an ensemble:
    if isinstance(path_to_weights, list):
        predictor_.ensemble_predict()

    elif isinstance(path_to_weights, str):
        predictor_.predict()

    else:
        raise ValueError(f"Argument 'path_to_weights' must be str or list of str, provided {type(path_to_weights)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Inference Script')
    parser.add_argument("-p", "--path", type=str, required=True, help="path to the config file")
    args = parser.parse_args()
    main(args)
