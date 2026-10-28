# LesionCover

LesionCover is a lesion-coverage crop selection rule for diabetic retinopathy grading. This repository holds the code for
choosing a few full-resolution crops of a color fundus photograph so that they cover as many distinct
predicted lesions as possible, and for grading diabetic retinopathy from a 512-pixel global view plus four 384-pixel crops.
A lesion segmenter (U-Net, ResNet-18) supplies calibrated lesion maps; crops are chosen by greedy coverage of lesion
components and compared with heatmap ranking, non-maximum suppression, a determinantal point process and fixed or random
crops; a ConvNeXt grader with gated attention pooling predicts the five-level grade. On photographs from other cameras,
predicted lesions that recur at the same position across a camera's unlabelled photographs (camera spots such as lens
dust) are left out of the coverage objective and inpainted before grading (`scripts/stages/05d_camera_prior.py`,
`07h_predict_camera.py`), without labels and with the trained networks unchanged.

## Data

Public datasets only: DDR (development and lesion masks), DeepDRiD, IDRiD, Messidor-2 and EyePACS (external testing) and
APTOS 2019 (additional training photographs in one setting). `scripts/stages/00_*` download them and record checksums.

## Usage

```bash
conda env create -f environment.yml && conda activate lesioncover
bash run_all.sh                     # all stages in order, then the figures; --force reruns, --skip=a,b skips stages
bash scripts/run_queue.sh config/grading_runs_all.txt <gpu>   # grading models in parallel, one lane per GPU
```

All parameters are in `config/run.yaml`. Each stage writes its tables to `results/<stage>/`; the scripts in
`scripts/figures/` read only these tables. The schematic icons of Fig. 1 are from Health Icons (healthicons.org, MIT
licence; `figures/source_data/icons_healthicons/`) or drawn in code.

## Licence

MIT
