# [DiffusionModelFamily](https://github.com/Hytidel/DiffusionModelFamily/)

[Hytidel](https://hytidel.github.io/)

[Tutorial Video (Chinese)](https://www.bilibili.com/video/BV1kB6EYcEHp/)

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FHytidel%2FDiffusionModelFamily%2F&count_bg=%23FF8383&title_bg=%23A19AD3&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

---

# 🧐 Introduction

😋 This project provides intuitive interfaces for [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev), [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base), [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo), [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), and [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo) with 🤗 [Diffusers](https://huggingface.co/diffusers). Users can easily perform basic text-to-image generation by modifying the YAML configuration files in the `./config` directory.

🤯 Additionally, this project serves as a simple template for deep learning with Python. Users can create their own tasks in a similar format and manage configuration files efficiently using the powerful [Hydra](https://hydra.cc/) library. 

---

# 💻 Installation

😀The installation is tested with NVIDIA Driver `550.67` , CUDA `11.8` and `setuptools==75.1.0` in Ubuntu `20.04.5 LTS`. 

## Environment

[1] Clone our repository from Github by HTTPS: 

```bash
git clone https://github.com/Hytidel/DiffusionModelFamily.git
```

​	or SSH: 

```bash
git clone git@github.com:Hytidel/DiffusionModelFamily.git
```

[2] Create a conda virtual environment with Python 3.10. 

```bash
conda create -n DiffusionModelFamily python=3.10 -y
```

[3] Install the dependencies. 

```bash
pip install -r requirement.txt
```

## Model Checkpoints

🥸 Below provides a guide on how to download model checkpoints using the `huggingface-cli` ([HuggingFace Command Line Interface](https://huggingface.co/docs/huggingface_hub/guides/cli)) library. Alternatively, you may use other methods to save the checkpoints locally on your machine.

[1] Install the `huggingface-cli` library within your conda environment.

```bash
pip install -U "huggingface_hub[cli]"
```

[2] (Optional) Customize the cache directory path.

```bash
echo "export HF_HOME=/root/autodl-tmp/cache/" >> ~/.bashrc
source ~/.bashrc
```

[3] (Optional) Configure a HuggingFace mirror site and enable acceleration.

```bash
echo "HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc
echo "HF_HUB_ENABLE_HF_TRANSFER=1" >> ~/.bashrc
source ~/.bashrc
```

[4] Download the model checkpoints. 

​	😶‍🌫️ The download is influenced by network conditions. If fails, retry several times or enable a proxy. 

​	(1) FLUX.1-dev

```bash
huggingface-cli download --resume-download --local-dir-use-symlinks False black-forest-labs/FLUX.1-dev --local-dir ~/autodl-tmp/black-forest-labs/FLUX.1-dev
```

​	(2) Stable Diffusion 2.1

```bash
huggingface-cli download --resume-download --local-dir-use-symlinks False stabilityai/stable-diffusion-2-1-base --local-dir ~/autodl-tmp/stabilityai/stable-diffusion-2-1-base
```

​	(3) SD-Turbo

```bash
huggingface-cli download --resume-download --local-dir-use-symlinks False stabilityai/sd-turbo --local-dir ~/autodl-tmp/stabilityai/sd-turbo
```

​	(4) Stable Diffusion XL

```bash
huggingface-cli download --resume-download --local-dir-use-symlinks False stabilityai/stable-diffusion-xl-base-1.0 --local-dir ~/autodl-tmp/stabilityai/stable-diffusion-xl-base-1.0
```

​	(5) SDXL-Turbo

```bash
huggingface-cli download --resume-download --local-dir-use-symlinks False stabilityai/sdxl-turbo --local-dir ~/autodl-tmp/stabilityai/sdxl-turbo
```

---

# 😇 Quick Start

🏞️ You can quickly explore the features by running the scripts in the `./script` directory. The output of each task will be stored in the `./tmp` directory within a subfolder named after the respective task. 

😄 Don't forget to check whether the script has executable permission. If not, grant it by running 

```bash
chmod +x /path/to/the/script
```



😎 We provide some scripts for you to run the tasks conveniently. 

* `./script/sample_flux.1-dev-cat.sh` : Generate an image with the text prompt "*A cat holding a sign that says hello world*" using FLUX.1-dev. 
* `./script/sample_sd_2.1-astronaut.sh` : Generate an image with the text prompt "*a photo of an astronaut riding a horse on mars*" using Stable Diffusion 2.1. 
* `./script/sample_sd-turbo-racoon.sh` : Generate an image with the text prompt "*A cinematic shot of a baby racoon wearing an intricate italian priest robe.*" using SD-Turbo. 
* `./script/sample_sdxl-lion.sh` : Generate an image with the text prompt "*A majestic lion jumping from a big stone at night*" using Stable Diffusion XL. 
* `./script/sample_sdxl-turbo-racoon.sh` : Generate an image with the text prompt "*A cinematic shot of a baby racoon wearing an intricate italian priest robe.*" using SDXL-Turbo. 
* `./script/sample_freeu-mouse.sh` : Generate an image based with the text prompt "*A photo of a cute mouse wearing a crown.*" using two models respectively: Stable Diffusion 2.1 and Stable Diffusion 2.1 with FreeU.



🤓 The format is `/path/to/the/script {gpu_id}` , where `gpu_id` specifies which GPU you would like this script to run on (if you have multi-GPU), `0` default. Remind that GPU indexing starts from `0`. 

🫣 For example, I can perform text-to-image with the text prompt "*a photo of an astronaut riding a horse on mars*" using Stable Diffusion 2.1 on GPU - 0 by

```bash
./script/sample_sd_2.1-astronaut.sh 0
```

---

# 🫡 Configuration

🤤 All configuration files are located in the `./config` directory, organized as follows: 

* Global settings are stored in `./config/cfg.yaml`.
* Pipeline-specific configurations are stored in YAML files under the `./config/pipeline` directory. 
* Task-specific configurations are stored in YAML files under the `./config/pipeline` directory. 



😝 To add a new task, implement it in the `./task` directory first, and invoke it in `./main.py`. 

---

# 🙏 Acknowledgement

* This Hydra-based project template is inspired by [RGBManip](https://github.com/hyperplane-lab/RGBManip).
* The implementation of the FreeU module is adapted from [FreeU](https://github.com/ChenyangSi/FreeU).



😘 We would like to thank the authors for their excellent works. 

---

# ⭐ Star
🥰 If you find our work helpful, please consider leaving a star ⭐.

---

# 🏷️ License
😌 This repository is released under the MIT License. 

---

---
