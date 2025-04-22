# MMEN

Official codes for paper "Transformer-based adaptive contrastive learning for multimodal sentiment analysis"

- [Dataset HomePage](https://thuiar.github.io/sims.github.io/chsims)

## 1. Data Download

1. CH-SIMS v2(s) - Supervised data:
  - [Google Drive](https://drive.google.com/drive/folders/1wFvGS0ebKRvT3q6Xolot-sDtCNfz7HRA?usp=sharing)
  - [Baiduyun Drive](https://pan.baidu.com/s/13Ds2_XDIGUqMHt4lXNLQSQ) Code: icmi

2. CH-SIMS v2(u) - Unsupervised data:
  - [Google Drive](https://drive.google.com/drive/folders/1llIbm3gwyJRwwk58RUQHWBNKjHI9vGGB?usp=sharing)
  - [Baiduyun Drive](https://pan.baidu.com/s/1tezEDR3Y23hJ6Mp5fmcp-w) Code: icmi 

## 2. Run Experiments

1. Download dataset and set correct path in:

```text
config/config_regression.py --> line 39  --> "root_dataset_dir"
```

2. If you want to run the MMEN framework: 

```shell
python run.py
```

## 3. Citation

If you find this paper or dataset useful, please cite us at: 

```bib
@article{hu2024transformer,
  title={Transformer-based adaptive contrastive learning for multimodal sentiment analysis},
  author={Hu, Yifan and Huang, Xi and Wang, Xianbing and Lin, Hai and Zhang, Rong},
  journal={Multimedia Tools and Applications},
  pages={1--18},
  year={2024},
  publisher={Springer}
}
```

## 6. Contact Us

For any questions, please email [Yihe Liu](mailto:512796310@qq.com) or [Ziqi Yuan](mailto:yzq21@mails.tsinghua.edu.cn)
