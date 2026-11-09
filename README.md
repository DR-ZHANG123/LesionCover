# LesionCover

LesionCover is a deep learning network for grading diabetic retinopathy (DR) from colour fundus photographs that keeps
**lesion evidence** apart from the **grading standard**:

- **Global tokens**: a DINOv2 ViT-L/14 encoder with fixed pretrained weights and rank-8 LoRA adapters encodes the
  1024-pixel global view (`scripts/lib/fmencoder.py`).
- **Lesion tokens**: a lesion segmenter (U-Net, ResNet-18) marks microaneurysms, haemorrhages, hard and soft exudates;
  each predicted lesion becomes a token with its type, confidence, size, position and appearance (`scripts/lib/lesiontokens.py`).
- **Quadrant rule**: lesion counts per quadrant pass through a differentiable rule with learnable thresholds per lesion
  type ("at least j quadrants reach threshold k"), which also gives a monotone rule-only ordinal grade whose thresholds
  and weights can be read after training (`scripts/lib/rulehead.py`).
- **Fusion and burden gate**: a Transformer fuses all tokens (`scripts/lib/lcnet.py`).
- **Standard layer**: five parameters (a temperature and four grade biases) that place the grade boundaries. At a new
  site, only these five parameters are re-estimated from a few labelled photographs per grade
  (`scripts/stages/08v_local_calibration.py`): train once, align with five photographs per grade.

The same lesion measurements serve as a common ruler to compare where cohorts place their grade boundaries
(`scripts/stages/08u_grading_standard.py`, `08w_ruler_check.py`), and the learned rule is read from the trained weights
(`08x_learned_rule.py`).

## Data

Public datasets only: DDR (training, development, test and lesion masks), DeepDRiD, IDRiD, Messidor-2, EyePACS and
MMRDR (external testing). `scripts/stages/00_*` download them and check checksums; sources are listed in
`metadata/data_sources.tsv`. GDRNet and CABNet are retrained with the official GDRBench code at a fixed commit
(`scripts/stages/07p_published_methods.py`).

## Usage

```bash
conda env create -f environment.yml && conda activate lesioncover
bash run_all.sh                     # all stages in order, then the figures; --force reruns, --skip=a,b skips stages
bash scripts/run_cmd_queue.sh config/lesioncover_runs.txt <gpu>   # train the networks in parallel, one lane per GPU
```

The final LesionCover network is
`python scripts/stages/07n_lcnet.py --seed <s> --tag fmrglearn --rule-learn --gate --fm vit_large_patch14_dinov2.lvd142m`,
trained with seeds 11, 22 and 33. All parameters are in `config/run.yaml`. Each stage writes its tables to
`results/<stage>/`; the scripts in `scripts/figures/` read only these tables.

## Licence

MIT
