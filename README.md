# CKD

ACL 2023 (Findings) paper: Distilling Calibrated Knowledge for Stance Detection.

## Abstract

Stance detection aims to determine the position of an author toward a target and provides insights into people{'}s views on controversial topics such as marijuana legalization. Despite recent progress in this task, most existing approaches use hard labels (one-hot vectors) during training, which ignores meaningful signals among categories offered by soft labels. In this work, we explore knowledge distillation for stance detection and present a comprehensive analysis. Our contributions are: 1) we propose to use knowledge distillation over multiple generations in which a student is taken as a new teacher to transfer knowledge to a new fresh student; 2) we propose a novel dynamic temperature scaling for knowledge distillation to calibrate teacher predictions in each generation step. Extensive results on three stance detection datasets show that knowledge distillation benefits stance detection and a teacher is able to transfer knowledge to a student more smoothly via calibrated guiding signals.

## Setup

You can download the project and install required packages using following commands:

```bash
git clone https://github.com/chuchun8/CKD.git
cd src
pip install -r requirements.txt
```

## Running

BERTweet is used as our baseline for in this paper. For our proposed calibrated knowledge distillation (CKD), run
```
bash train.sh > ckd.log
```
Specifically, in train.sh,
`-calib` indicates that we run with CKD (temperature is dynamically updated in each generation).

`-anneal` indicates that we run our experiments with teacher annealing.

`-t` indicates the default temperature for knowledge distillation baselines (not used in CKD). For example, `-t 2` and no `-calib` mean that we run normal knowledge distillation with fixed temperature 2 (KD-2).

## Contact Info

Please contact Yingjie Li at liyingjie@westlake.edu.cn or yli300@uic.edu with any questions.

## Citation

```bibtex
@inproceedings{li-caragea-2023-distilling,
    title = "Distilling Calibrated Knowledge for Stance Detection",
    author = "Li, Yingjie  and
      Caragea, Cornelia",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.393",
    pages = "6316--6329",
}
```
