<div align="center">
  <h1>Multi-Level Correlation Network For Few-Shot Image Classification <br> (ICME 2023)</h1>
</div>

<div align="center">
  <h4> <a href=https://ieeexplore.ieee.org/abstract/document/10219908>[paper]</a></h4>
</div>


## :heavy_check_mark: Requirements
* Ubuntu 16.04
* Python 3.7
* [CUDA 11.0](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 1.7.1](https://pytorch.org)


## :gear: Conda environmnet installation
```bash
conda env create --name mlcn --file environment.yml
conda activate mlcn
```

## :books: Datasets
```bash
cd datasets
bash download_miniimagenet.sh
bash download_cub.sh
bash download_cifar_fs.sh
bash download_tieredimagenet.sh
```


## :mag: Related repos
Our project references the codes in the following repos:

* Zhang _et al_., [DeepEMD](https://github.com/icoz69/DeepEMD).
* Ye _et al_., [FEAT](https://github.com/Sha-Lab/FEAT)
* Kang _et al_., [renet](https://github.com/dahyun-kang/renet)

## :love_letter: Acknowledgement
This paper adopted the main code bases from [renet](https://github.com/dahyun-kang/renet).

I also sincerely thank Min Zhang. 


## :scroll: Citing MLCN
If you find our code or paper useful to your research work, please consider citing our work using the following bibtex:
```
@inproceedings{dang2023multi,
  title={Multi-Level Correlation Network For Few-Shot Image Classification},
  author={Dang, Yunkai and Sun, Meijun and Zhang, Min and Chen, Zhengyu and Zhang, Xinliang and Wang, Zheng and Wang, Donglin},
  booktitle={2023 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={2909--2914},
  year={2023},
  organization={IEEE}
}
```
