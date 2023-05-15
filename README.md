# GRVCTrack

This repo provides an implementation of the BYTE Tracker along with pretrained yolo model
## Installation:

create a virtual env with python3.9 \n
conda env create -n GRVCTrack python=3.9
conda activate GRVCTrack
git clone https://github.com/Polbv/GRVCTrack
cd GRVCTrack
pip install -r requirements.txt
## MODEL ZOO
Pretrained models must be provided in the form of a .cfg and .weights
link to pretrained weights: https://drive.google.com/drive/folders/1tgmGgsx6nOu_I6XlDDUvsjOJjch4bcFU?usp=sharing





## Citation

```
@article{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## Acknowledgement

A large part of the code is borrowed from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [FairMOT](https://github.com/ifzhang/FairMOT), [TransTrack](https://github.com/PeizeSun/TransTrack) and [JDE-Cpp](https://github.com/samylee/Towards-Realtime-MOT-Cpp). Many thanks for their wonderful works.
