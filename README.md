# Completer: Incomplete Multi-view Clustering via Contrastive Prediction

This repo contains the code and data of the following paper accepeted by [CVPR 2021](http://cvpr2021.thecvf.com)

> COMPLETER: Incomplete Multi-view Clustering via Contrastive Prediction

![image](https://github.com/Lin-Yijie/2021-CVPR-Completer/blob/main/figs/framework.png)

## Requirements

pytorch 1.2.0 

numpy>=1.16.4

## Configuration

The hyper-parameters and the missing rate are defined in configure.py.

## Usage

The code includes:

- an example implementation of the model,
- an example clustering task for different missing rates.

```bash
python run.py --dataset 0 --devices 0 --print_num 100 --test_time 5
```

You can get the following output:

```bash
Epoch : 100/500 ===> Reconstruction loss = 0.2787===> Reconstruction loss = 0.0273 ===> Dual prediction loss = 0.0166  ===> Contrastive loss = -4.4762e+02 ===> Loss = -4.4759e+02
view_concat {'kmeans': {'AMI': 0.5969, 'NMI': 0.6106, 'ARI': 0.6044, 'accuracy': 0.5813, 'precision': 0.4408, 'recall': 0.3835, 'f_measure': 0.3921}}
Epoch : 200/500 ===> Reconstruction loss = 0.2609===> Reconstruction loss = 0.0228 ===> Dual prediction loss = 0.0012  ===> Contrastive loss = -4.4805e+02 ===> Loss = -4.4802e+02
view_concat {'kmeans': {'AMI': 0.6575, 'NMI': 0.6691, 'ARI': 0.6974, 'accuracy': 0.6593, 'precision': 0.4551, 'recall': 0.4222, 'f_measure': 0.4096}}
Epoch : 300/500 ===> Reconstruction loss = 0.2472===> Reconstruction loss = 0.0232 ===> Dual prediction loss = 0.0010  ===> Contrastive loss = -4.4864e+02 ===> Loss = -4.4861e+02
view_concat {'kmeans': {'AMI': 0.6875, 'NMI': 0.6982, 'ARI': 0.8679, 'accuracy': 0.7439, 'precision': 0.4586, 'recall': 0.444, 'f_measure': 0.4217}}
Epoch : 400/500 ===> Reconstruction loss = 0.2357===> Reconstruction loss = 0.0214 ===> Dual prediction loss = 0.0013  ===> Contrastive loss = -4.4816e+02 ===> Loss = -4.4813e+02
view_concat {'kmeans': {'AMI': 0.692, 'NMI': 0.7027, 'ARI': 0.8736, 'accuracy': 0.7456, 'precision': 0.4601, 'recall': 0.4451, 'f_measure': 0.4257}}
Epoch : 500/500 ===> Reconstruction loss = 0.2298===> Reconstruction loss = 0.0198 ===> Dual prediction loss = 0.0016  ===> Contrastive loss = -4.4796e+02 ===> Loss = -4.4793e+02
view_concat {'kmeans': {'AMI': 0.6912, 'NMI': 0.7019, 'ARI': 0.8707, 'accuracy': 0.7464, 'precision': 0.4657, 'recall': 0.4464, 'f_measure': 0.4265}}
```

## Citation

If you find our work useful in your research, please consider citing:

```latex
@inproceedings{lin2021completer,
   title={COMPLETER: Incomplete Multi-view Clustering via Contrastive Prediction},
   author={Yijie Lin, Yuanbiao Gou, Zitao Liu, Boyun Li, Jiancheng Lv, Xi Peng},
   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
   month={June},
   year={2021}
}
```
