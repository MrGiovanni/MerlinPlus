# MerlinPlus
[MICCAI 2026] Merlin Plus

Planned release:

- **AI-made organ segmentation masks** (used to train R-Super): **released** at https://huggingface.co/datasets/AbdomenAtlas/MerlinPlus
- **Longitudinal metadata**: **released** in [merlin_longitudinal_metadata.csv](merlin_longitudinal_metadata.csv) in this repository. This metadata provides de-identified patient IDs and examination dates, enabling you to identify CT scans from the same patient and calculate the time interval between examinations. It therefore supports longitudinal model training and evaluation. We additionally provide de-identified patient age, race, sex, scanner manufacturer and model, CT voxel spacing, contrast status, contrast phase, kVp, and X-ray tube current.
- **Radiologist-made tumor segmentation masks**: soon

## Paper:
Coming soon!

## Citation:
If you use the code, data or methods in this repository, please cite all papers below:

```
@article{bassi2025scaling,
  title={Scaling Artificial Intelligence for Multi-Tumor Early Detection with More Reports, Fewer Masks},
  author={Bassi, Pedro RAS and Zhou, Xinze and Li, Wenxuan and P{\l}otka, Szymon and Chen, Jieneng and Chen, Qi and Zhu, Zheren and Prz{\k{a}}do, Jakub and Hamac{\i}, Ibrahim E and Er, Sezgin and others},
  journal={arXiv preprint arXiv:2510.14803},
  year={2025}
}

@article{blankemeier_kumar2026merlin,
  author = {Blankemeier, Louis and Kumar, Ashwin and Cohen, Joseph Paul and Liu, Jiaming and Liu, Longchao and Van Veen, Dave and Gardezi, Syed Jamal Safdar and Yu, Hongkun and Paschali, Magdalini and Chen, Zhihong and Delbrouck, Jean-Benoit and Reis, Eduardo and Holland, Robbie and Truyts, Cesar and Bluethgen, Christian and Wu, Yufu and Lian, Long and Jensen, Malte Engmann Kjeldskov and Ostmeier, Sophie and Varma, Maya and Valanarasu, Jeya Maria Jose and Fang, Zhongnan and Huo, Zepeng and Nabulsi, Zaid and Ardila, Diego and Weng, Wei-Hung and Amaro Junior, Edson and Ahuja, Neera and Fries, Jason and Shah, Nigam H. and Zaharchuk, Greg and Willis, Marc and Yala, Adam and Johnston, Andrew and Boutin, Robert D. and Wentland, Andrew and Langlotz, Curtis P. and Hom, Jason and Gatidis, Sergios and Chaudhari, Akshay S.},
  title   = {Merlin: a computed tomography vision-language foundation model and dataset},
  journal = {Nature},
  year    = {2026},
  doi     = {10.1038/s41586-026-10181-8},
  url     = {https://doi.org/10.1038/s41586-026-10181-8}
}
```



## Acknowledgement

This work was supported by the McGovern Foundation and the Lustgarten Foundation for Pancreatic Cancer Research. Paper content is covered by patents pending.
