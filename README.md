# Plain-Det

By [Cheng Shi](https://chengshiest.github.io/), Yuchen Zhu and
[Sibei Yang](https://faculty.sist.shanghaitech.edu.cn/yangsibei/)

The official PyTorch implementation of the "Plain-Det: A Plain Multi-Dataset Object Detector".

# Main results
| BoxRPB | MIM PT. | Reparam. | AP | Paper Position | CFG | CKPT |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| ✗ | ✗ | ✗ | 37.2 | Tab2 Exp1 | [cfg](./configs/swinv2_small_sup_pt_ape.sh) | [ckpt](https://msravcghub.blob.core.windows.net/plaindetr-release/plaindetr_models/plaindetr_swinv2_small_sup_pt_ape.pth?sv=2020-04-08&st=2023-11-11T09%3A53%3A39Z&se=2049-12-31T09%3A53%3A00Z&sr=b&sp=r&sig=1ntSPDFHXBDVfYts8aUX8xTxA2kRpZyEQdwZL0tRNSk%3D)
| ✓ | ✗ | ✗ | 46.1 | Tab2 Exp2 | [cfg](./configs/swinv2_small_sup_pt_boxrpe.sh) | [ckpt](https://msravcghub.blob.core.windows.net/plaindetr-release/plaindetr_models/plaindetr_swinv2_small_sup_pt_boxrpe.pth?sv=2020-04-08&st=2023-11-11T09%3A54%3A10Z&se=2023-11-12T09%3A54%3A10Z&sr=b&sp=r&sig=yg4gw7vWX8zlOurS3x2J9%2BPwfsSaHEYYOHE4DJTWw%2BQ%3D) 
| ✓ | ✓ | ✗ | 48.7 | Tab2 Exp5 | [cfg](./configs/swinv2_small_mim_pt_boxrpe.sh) | [ckpt](https://msravcghub.blob.core.windows.net/plaindetr-release/plaindetr_models/plaindetr_swinv2_small_mim_pt_boxrpe.pth?sv=2020-04-08&st=2023-11-11T09%3A52%3A38Z&se=2049-12-31T09%3A52%3A00Z&sr=b&sp=r&sig=eX%2FNgca78ccyBhlujtCSh1BDHiPPOjjceyrMMLxKgr8%3D)
| ✓ | ✓ | ✓ | 50.9 | Tab2 Exp6 | [cfg](./configs/swinv2_small_mim_pt_boxrpe_reparam.sh) | [ckpt](https://msravcghub.blob.core.windows.net/plaindetr-release/plaindetr_models/plaindetr_swinv2_small_mim_pt_boxrpe_reparam.pth?sv=2020-04-08&st=2023-11-14T07%3A42%3A25Z&se=2049-12-31T07%3A42%3A00Z&sr=b&sp=r&sig=5r09k4tFNIO%2FURYIQ2RbjJOU7v4dWqFW1D3F%2Bdg%2FYq0%3D)
# Installation

# Usage

# Citing Plain-Det

If you find Plain-Det useful in your research, please consider citing:
```
inproceedings{
  shi2024plain,
  title={Plain-Det: A Plain Multi-Dataset Object Detector},
}
```