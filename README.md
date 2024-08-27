# [HuMoUr](https://github.com/HariWu1995/HuMoUr): Human-Motion Wrapper for the below repos

✅ [MDM](https://github.com/GuyTevet/motion-diffusion-model)

✅ [PriorMDM](https://github.com/priorMDM/priorMDM)

✅ [SinMDM](https://github.com/SinMDM/SinMDM)

⬜ [MAS](https://github.com/roykapon/MAS)

✅ [MoMo](https://github.com/MonkeySeeDoCG/MoMo)

⬜ [CAMDM](https://github.com/AIGAnimation/CAMDM)

⬜ [THOI](https://github.com/JunukCha/Text2HOI)

⬜ [HOID](https://github.com/neu-vi/HOI-Diff)


## Getting started

This code was tested on `Kaggle` and requires:

* Python 3.10
* GPU T4 x2


### 0. Setup environment

Install ffmpeg:

```shell
sudo apt update
sudo apt install ffmpeg
```

Setup environment:
```shell
!pip install -q git+https://github.com/openai/CLIP.git
!pip install -q blis==0.7.8 chumpy==0.70 click==8.1.3 confection==0.0.2 \
                importlib-metadata==5.0.0 lxml==4.9.1 srsly==2.4.4 \
                murmurhash==1.0.8 preshed==3.0.7 pycryptodomex==3.15.0 \
                thinc==8.0.17 wasabi==0.10.1 wcwidth==0.2.5 \
                regex==2022.9.13 spacy==3.3.1 \
                ftfy==6.1.1 einops \
                mkl==2021.4.0 mkl_fft mkl-service mkl_random
!pip install -q trimesh torchgeometry "smplx[all]"
!pip install -q numpy==1.23.5
!pip install -q matplotlib==3.1.3

!python -m spacy download en_core_web_sm
```

### 1. Download dependencies:

<details>
  <summary><b>Samples</b></summary>

```bash
bash prepare/download_smpl_files.sh
```
</details>

<details>
  <summary><b>Text to Motion</b></summary>

```bash
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```
</details>

<details>
  <summary><b>Action to Motion</b></summary>

```bash
bash prepare/download_recognition_models.sh
```
</details>

<details>
  <summary><b>Unconstrained</b></summary>

```bash
bash prepare/download_recognition_unconstrained_models.sh
```
</details>


### 2. Get data

Run those notebooks, or just get output:

[MDM Datasets](https://www.kaggle.com/code/mrriandmstique/download-humour-dataset)

[Prior-MDM Datasets](https://www.kaggle.com/mrriandmstique/download-humour-datasets-2)


### 3. Get checkpoints

Run this notebook, or just get output:

[MDM + prior-MDM checkpoints](https://www.kaggle.com/code/mrriandmstique/download-humour-checkpoints)


### 4. Motion Synthesis

Run this notebook, or copy and test with your customization:

[HuMoUr Test](https://www.kaggle.com/code/mrriandmstique/humorapid-test)


### 5. API

* ![On Progress](https://progress-bar.dev/10/?scale=100&title=HuMoRapiD&suffix=%&width=200&color=babaca)
