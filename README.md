# Learning to Render Novel Views from Wide-Baseline Stereo Pairs 
### [Project Page](https://yilundu.github.io/wide_baseline/) | [Paper](https://arxiv.org/abs/2304.08463) 
[![Explore in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PeL5oJ_eraLEdzTEVPLBwoM2pyv26WcU?usp=sharing)<br>

[Yilun Du](https://yilundu.github.io/),
[Cameron Smith](https://scholar.google.com/citations?user=zrZNo3wAAAAJ&hl=en&oi=sra),
[Ayush Tewari](https://ayushtewari.com/),
[Vincent Sitzmann](https://vsitzmann.github.io/)
<br>
MIT

This is a official implementation of the paper "Learning to Render Novel Views from Wide-Baseline Stereo Pairs". 


## Google Colab
If you want to experiment with our approach, we have written a [Colab](https://colab.research.google.com/drive/1PeL5oJ_eraLEdzTEVPLBwoM2pyv26WcU?usp=sharing).
It loads two input images and estimates the underlying fundemental matrix between images. It doesn't require 
installing anything, and illustrates and renders videos when interpolating between the two given images.

## Get started
You can install the packages used in this codebase use the following command
```
pip install -r requirements.txt
```
You will need to download the Realestate10k and ACID dataset. We have attached a small subset of the RealEstate10K dataset as well as pose files which we may directly download using the dropbox link [here](https://www.dropbox.com/s/qo8b7odsms722kq/cvpr2023_wide_baseline_data.tar.gz?dl=0).
Please extract the folder in the root directory of the repo.

To download the full datasets for Realestate10k and ACID, please follow the README [here](./data_download/README.md).

You can also load a pretrained model on the RealEstate dataset [here](https://drive.google.com/file/d/1hxiyjWYR1UOOcuxTHZw7_B5VNqynmC5f/view). Please extract the model in the root directory of the repo.

## High-Level structure
The code is organized as follows:
* ./dataset/ contains code loading data
* ./data_download/ contains code to download Realestate10k and ACID datasets
* ./utils/ contains code for different utility functions on the dataset
* ./estimate_pose/ contains code for estimate the pose between two images using superpoint
* models.py contains the underlying code for instantiating the model
* training.py contains the underlying code for training the model
* loss_functions.py contains loss functions for the different experiments.
* summaries.py contains summary functions for training
* ./experiment_scripts/ contains scripts to reproduce experiments in the paper.

## Reproducing experiments
The directory `experiment_scripts` contains one script per experiment in the paper.

To monitor progress, the training code writes tensorboard summaries into a "summaries"" subdirectory in the logging_root. The code will run 
infinitely -- you may stop the code after a day or two of training.

### Realestate experiments
The Realestate experiment can be reproduced by running the command below for two days
```
python experiment_scripts/train_realestate10k.py --experiment_name realestate --batch_size 12 --gpus 4
```

followed by running the command below (adding lpips and depth loss later in training)
```
python experiment_scripts/train_realestate10k.py --experiment_name realestate_lpips_depth --batch_size 4 --gpus 4 --lpips --depth --checkpoint_path logs/realestate/checkpoints/model_current.pth
```

You can evaluate the results of the trained model using
```
python experiment_scripts/eval_realestate10k.py --experiment_name vis_realestate --batch_size 12 --gpus 1 --views=2 --checkpoint_path logs/realestate/checkpoints/model_current.pth
```

You can also visualize the results of applying the trained model on videos using the command
```
python experiment_scripts/render_realestate10k_traj.py --experiment_name vis_realestate --batch_size 12 --gpus 1 --checkpoint_path logs/realestate/checkpoints/model_current.pth
```

and on unposed images using

```
python experiment_scripts/render_unposed_traj.py --experiment_name vis_realestate --batch_size 12 --gpus 1 --checkpoint_path logs/realestate/checkpoints/model_current.pth
```

## Citation
If you find our work useful in your research, consider cite:
```
@article{du2023widerender,
          title={Learning to Render Novel Views from
            Wide-Baseline Stereo Pairs},
          author={Du, Yilun and Smith, Cameron and Tewari,
                Ayush and Sitzmann, Vincent},
          journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
          year={2023}
}
```

## Contact
If you have any questions, please feel free to email the authors.
