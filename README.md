# Multi-direction and Multi-scale Pyramid in Transformer for Video-based Pedestrian Retrieval on PolarbearVidID
![LICENSE](https://img.shields.io/badge/license-GPL%202.0-green) ![Python](https://img.shields.io/badge/python-3.6-blue.svg) ![pytorch](https://img.shields.io/badge/pytorch-1.8.1-orange) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-direction-and-multi-scale-pyramid-in-1/person-re-identification-on-ilids-vid)](https://paperswithcode.com/sota/person-re-identification-on-ilids-vid?p=multi-direction-and-multi-scale-pyramid-in-1) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-direction-and-multi-scale-pyramid-in-1/person-re-identification-on-mars)](https://paperswithcode.com/sota/person-re-identification-on-mars?p=multi-direction-and-multi-scale-pyramid-in-1)

Implementation of the proposed PiT used on the PolarBearVidID Dataset. Please refer to [[PolarBearVidID @ MDPI]](https://www.mdpi.com/2076-2615/13/5/801) and [[PiT @ Arxiv]](https://arxiv.org/pdf/2202.06014.pdf).

![dataset](./Dataset.png)
![framework](./framework.jpg)


## Getting Started
### Requirements
Here is a brief instruction for installing the experimental environment.
```
# Windows 10 and 11 (use cmd)
# install conda (add to path)
$ conda create -n PiT python=3.6 -y
$ conda activate PiT (Win 11: activate PiT)
# install pytorch 1.8.1/1.6.0 (other versions may also work)
$ pip install timm scipy einops yacs opencv-python==4.3.0.36 tensorboard pandas
```

### Download pre-trained model
The pre-trained vit model can be downloaded in this [link](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) and should be put in the `checkpoints/` directory.

### Dataset Preparation
Download PolarBearVidID Dataset at [PolarBearVidID](https://zenodo.org/record/7564529) and store it in the `data/PolarBearVidID` Directory.

## Training and Testing
```
# This command below includes the training and testing processes.
$ python train.py --config_file configs/PolarBearVidID/pit.yml MODEL.DEVICE_ID "('0')"
# For testing only
$ python train.py --config_file configs/PolarBearVidID/pit-test.yml MODEL.DEVICE_ID "('0')"
```

## Visualize findings
```
$ tensorboard --logdir_spec fold1:logs\PolarBearVidID_PiT\1,fold2:logs\PolarBearVidID_PiT\2,fold3:logs\PolarBearVidID_PiT\3,fold4:logs\PolarBearVidID_PiT\4,fold5:logs\PolarBearVidID_PiT\5
```


## Results in the Paper
TODO
2023-07-05 23:37:19,664 pit INFO: 5 trails average:
2023-07-05 23:37:19,665 pit INFO: mAP: 18.366%
2023-07-05 23:37:19,665 pit INFO: CMC curve, Rank-1  :12.274%
2023-07-05 23:37:19,665 pit INFO: CMC curve, Rank-5  :29.364%
2023-07-05 23:37:19,665 pit INFO: CMC curve, Rank-10 :47.281%
2023-07-05 23:37:19,665 pit INFO: CMC curve, Rank-20 :59.931%
TODO

The results of MARS and iLIDS-VID are trained using one 24G NVIDIA GPU and provided below. You can change the parameter `DATALOADER.P` in yml file to decrease the GPU memory cost.

| Model | Rank-1@MARS | Rank-1@iLIDS-VID |
| --- | --- | --- |
| PiT |  [90.22](https://pan.baidu.com/s/1nw5yofEilW0ffG_ZF4eoXQ) (code:wqxv)|  [92.07](https://pan.baidu.com/s/10LosWwUMktTiWvbHEP1Tjw) (code: quci)|

You can download these models and put them in the `../logs/[DATASET]_PiT_1x210_3x70_105x2_6p` directory. Then use the command below to evaluate them.
 ```
$ python test.py --config_file configs/PolarBearVidID/pit.yml MODEL.DEVICE_ID "('0')" 
```


## Acknowledgement

This repository is built upon the repository [TranReID](https://github.com/damo-cv/TransReID) and [PiT](https://github.com/deropty/PiT).

## Citation
If you find this project useful for your research, please kindly cite:
```
@article{zuerl_polarbearvidid_2023,
	title = {{PolarBearVidID}: {A} {Video}-{Based} {Re}-{Identification} {Benchmark} {Dataset} for {Polar} {Bears}},
	volume = {13},
	issn = {2076-2615},
	shorttitle = {{PolarBearVidID}},
	url = {https://www.mdpi.com/2076-2615/13/5/801},
	doi = {10.3390/ani13050801},
	language = {en},
	number = {5},
	urldate = {2023-02-27},
	journal = {Animals},
	author = {Zuerl, Matthias and Dirauf, Richard and Koeferl, Franz and Steinlein, Nils and Sueskind, Jonas and Zanca, Dario and Brehm, Ingrid and Fersen, Lorenzo von and Eskofier, Bjoern},
	month = feb,
	year = {2023},
	pages = {801}
}
```
And the original authors of PiT

```
@ARTICLE{9714137,
  author={Zang, Xianghao and Li, Ge and Gao, Wei},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={Multidirection and Multiscale Pyramid in Transformer for Video-Based Pedestrian Retrieval}, 
  year={2022},
  volume={18},
  number={12},
  pages={8776-8785},
  doi={10.1109/TII.2022.3151766}
}
```

## License
This repository is released under the GPL-2.0 License as found in the [LICENSE](LICENSE) file.
