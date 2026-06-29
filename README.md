# MerlinPlus
[MICCAI 2026] Merlin Plus

coming in June 2026.

Planned release:

- **AI-made organ segmentation masks** (used to train R-Super): **released**, https://huggingface.co/datasets/AbdomenAtlas/MerlinPlus
- **Longitudinal metadata**: **released** in merlin_longitudinal_metadata.csv in this repository. This metadata provides de-identified patient IDs and examination dates, enabling you to identify CT scans from the same patient and calculate the time interval between examinations. It therefore supports longitudinal model training and evaluation. We additionally provide de-identified patient age, race, sex, scanner manufacturer and model, CT voxel spacing, contrast status, contrast phase, kVp, and X-ray tube current.
- **Radiologist-made tumor segmentation masks**: soon
