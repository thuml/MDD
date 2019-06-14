# Margin Disparity Discrepancy

## Prerequisites:

* Python3
* PyTorch ==0.3.1 (with suitable CUDA and CuDNN version)
* torchvision == 0.2.0
* Numpy
* argparse
* PIL
* tqdm

## Dataset:

You need to modify the path of the image in every ".txt" in "./data".

## Training:

You can run "./scripts/train.sh" to train and evaluate on the task. Before that, you need to change the project root, dataset (Office-Home or Office-31), data address and CUDA_VISIBLE_DEVICES in the script.

## Citation:

If you use this code for your research, please consider citing:

```
@inproceedings{MDD_ICML_19,
  title={Bridging Theory and Algorithm for Domain Adaptation},
  author={Zhang, Yuchen and Liu, Tianle and Long, Mingsheng and Jordan, Michael},
  booktitle={International Conference on Machine Learning},
  pages={7404--7413},
  year={2019}
}
```
## Contact
If you have any problem about our code, feel free to contact zhangyuc17@mails.tsinghua.edu.cn.
