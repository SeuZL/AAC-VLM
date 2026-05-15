# 👁️ AAC-VLM: Agent-Anchored Collaborative Framework for Multi-Image Clinical Reports

AAC-VLM is an agent-anchored collaborative framework designed for multi-image clinical decision support (e.g., fundus fluorescein angiography, FFA). By introducing an ophthalmology-specialized agent (EvidenceAnchorAgent, EAA) as a visual-semantic anchor, we fully decouple image understanding from strictly formatted report writing, effectively mitigating the “format-induced hallucination” commonly observed in vision-language models (VLMs) under strong formatting constraints.

## 📑 Table of Contents

- [🌟 Key Highlights]
- [🏗️ Framework Overview]
- [🛠️ Environment Setup]
- [📂 Data Preparation]
- [📦 Model Weights]
- [🚀 Quick Start]
- [📝 Full Pipeline]
- [🤝 Acknowledgements]

## 🌟 Key Highlights

- Agent-anchored collaboration: We introduce, for the first time, a lightweight specialist agent (EAA) to reconstruct a reliable visual evidence chain in multi-image workflows.
- Reduced format-induced hallucination: With a “two-stage structured prompting” scheme and a “frame-wise generation + cross-frame summarization” strategy, image reading and report writing are fully decoupled.
- Clinical-grade reliability: On real-world FFA tests, 70.18% of generated reports were rated as directly clinically usable in blinded expert reviews.

## 🏗️ Framework Overview

> AAC-VLM contains three core components:
1) Agent-assisted sentence-level filtering;

2) Two-stage structured prompt construction;

3) Two-stage report generation and cross-frame summarization based on a backbone VLM.

## 🛠️ Environment Setup

This project is developed on Ascend 910 NPU and supports the PyTorch ecosystem.

### Clone the repository

```bash
git clone https://github.com/SeuZL/Agent-Anchored_Collaborative_VLM.git
cd Agent-Anchored_Collaborative_VLM
```

### Create a virtual environment

```bash
conda create -n aac-vlm python=3.10 -y
conda activate aac-vlm
```

### Install dependencies

```bash
pip install torch==2.4.0 torch_npu==2.4.0.post2 torchvision==0.19.0
pip install transformers ms-swift deepspeed
pip install -r requirements.txt
```

## 📂 Data Preparation

Organize your dataset in the following structure. Public benchmark datasets (FFA-IR and MM-Retinal v1) can be obtained from their official sources. Internal clinical data are not publicly released due to ethical restrictions.

```text
data/
├── ffa_ir/
│ ├── images/
│ └── annotations.jsonl
├── mm_retinal_v1/
└── custom_data/
├── images/
└── labels.jsonl
```

Download links:

- FFA-IR: https://physionet.org/content/ffa-ir-medical-report/1.1.0/
- MM-Retinal v1: https://drive.google.com/drive/folders/177RCtDeA6n99gWqgBS_Sw3WT6qYbzVmy

## 📦 Model Weights

- EvidenceAnchorAgent (EAA)
- Backbone: EfficientNet-B0
- Link: https://huggingface.co/timm/efficientnet_b0.ra_in1k
- AAC-VLM Base
- Backbone: Qwen2.5-VL-32B-Instruct
- Link: https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct

## 🚀 Quick Start

The dataset is “multi-image, single-report”. First, duplicate the report for each image and convert it into a one-to-one jsonl format.

Agent-assisted sentence-level filtering

Use EAA to extract clinical tags from images.

Example jsonl:

### Example.jsonl:

```json
{"messages": [{"role": "user", "content": "You are an ophthalmologist proficient in fundus diseases. Carefully read the image and write a textual description.", "image": "/data/FFApic2/001583660002.png"}, {"role": "assistant", "content": "Macular vessels are tortuous; a WISS ring is visible superior to the optic disc."}]}
```

### Edit the file paths and run:

```bash
python fortagdata.py
```

The model will append tags for all images in the dataset.

Example output:

### Example_taged.jsonl:

```json
{"messages": [{"role": "user", "content": "You are an ophthalmologist proficient in fundus diseases. Carefully read the image and write a textual description.", "image": "/data/FFApic2/001583660002.png"}, {"role": "assistant", "content": "Macular vessels are tortuous; a WISS ring is visible superior to the optic disc."}], "tag": ["Macula", "Fovea", "Optic disc", "Artery", "Vein", "Capillary", "Posterior pole", "Perifoveal ring"]}
```

### Edit paths and run:

```bash
python selecttag.py
```

You will get the filtered dataset:

filtered_Example.jsonl

VLM report generation

Using ms-swift inference framework, organize the dataset and replace the prompt with the Stage-1 prompt.

### Stage-1 prompt example:

```json
{"messages": [{"role": "user", "content": "You are an ophthalmologist proficient in fundus diseases. You are given a right-eye FA image. Patient: female, 51 years old. Series Time: 142325, Acquisition Time: 142446. The model predicts the image may contain the following tags: “Macula, Fovea, Optic disc, Artery, Vein, Capillary, Posterior pole, Perifoveal ring”. Carefully read the image and write an interpretation report.", "image": "/data/FFApice2/001583660002.png"}, {"role": "assistant", "content": "Macular vessels are tortuous; a WISS ring is visible superior to the optic disc."}]}
```

### Run:

```bash
NPROC_PER_NODE=4 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=802816 \
swift infer \
--adapters /data/output/checkpoint-2161 \
--val_dataset /root/filtered_Example_prompt1.jsonl \
--infer_backend pt \
--max_batch_size 1 \
--device_map auto \
--temperature 0.05 \
--repetition_penalty 1 \
--top_p 0.95 \
```

--max_new_tokens 512

Re-organize the training dataset using the Stage-1 outputs (OutputExample) and replace the prompt with the Stage-2 prompt.

Stage-2 prompt example (standardization rules omitted here for brevity; keep the same rules as in the Chinese version):

```json
{"messages": [{"role": "user", "content": "... Standardization rules ... Other information the image may contain: OutputExample", "image": "/data/FFApice2/001583660002.png"}, {"role": "assistant", "content": "Macular vessels are tortuous; a WISS ring is visible superior to the optic disc."}]}
```

### Run:

```bash
NPROC_PER_NODE=4 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=802816 \
swift infer \
--adapters /data/output/checkpoint-2161 \
--val_dataset /root/filtered_Example_prompt2.jsonl \
--infer_backend pt \
--max_batch_size 1 \
--device_map auto \
--temperature 0.05 \
--repetition_penalty 1 \
--top_p 0.95 \
```

--max_new_tokens 512

## 📝 Full Pipeline

Environment setup

### - Clone:

```bash
git clone https://github.com/SeuZL/Agent-Anchored_Collaborative_VLM.git
cd Agent-Anchored_Collaborative_VLM
```

### - Create env:

```bash
conda create -n aac-vlm python=3.10 -y
conda activate aac-vlm
```

### - Install deps (fill in exact versions if needed):

```bash
pip install torch==<TODO> torchvision==<TODO>
pip install transformers ms-swift deepspeed
pip install -r requirements.txt
```

Train EAA

- Prepare the dataset (see multi_label_data.jsonl as an example)
### - Run:

```bash
python trainclassify.py
```

Filter training data

### - After obtaining the best EAA checkpoint:

```bash
python fortagdata.py
python selecttag.py
```

- You will obtain the filtered dataset.

### Train the foundation model (FM)

```bash
nohup bash -c 'export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HCCL_CHECK_TIMEOUT=600 MAX_PIXELS=802816 TORCH_NPU_FUSION_ENABLE=1; torchrun --nproc_per_node=8 --rdzv_conf "overlap_timeout=600" /root/zhangliang/Qwen2.5-VL/Lora/ms-swift/swift/cli/sft.py --ddp_backend hccl --model /root/Qwen2.5vl-32B-Instruct --dataset /root/Qwen2.5-VL/yanke/ori_selected_withtag_newprompt.jsonl --deepspeed /root/Qwen2.5-VL/yanke/deepspeedzero3.json --train_type lora --torch_dtype bfloat16 --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --learning_rate 1e-5 --lr_scheduler_type cosine --warmup_ratio 0.1 --weight_decay 0.01 --lora_rank 32 --lora_alpha 64 --lora_dropout 0.1 --use_rslora False --max_grad_norm 1.0 --target_modules all-linear --freeze_vit true --eval_steps 100 --save_steps 100 --save_total_limit 100 --logging_steps 50 --max_length 8192 --output_dir /data/output/yanke --dataloader_num_workers 8 --model_kwargs "{\"device_map\":{\"\": \"npu:auto\"}}" --optim adamw_hf --gradient_checkpointing True' > /data/logs/qwen2.5lora/alldata32B.txt 2>&1 &
```

Inference

### Option A: use ms-swift demo:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift export \
--adapters /data/checkpoint-2161 \
--merge_lora true \
```

--device_map auto

Option B: run inference on the organized dataset (see infer_example.jsonl):

```bash
NPROC_PER_NODE=4 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=802816 \
swift infer \
--adapters /data/output/checkpoint-2161 \
--val_dataset /root/filtered_Example.jsonl \
--infer_backend pt \
--max_batch_size 1 \
--device_map auto \
--temperature 0.05 \
--repetition_penalty 1 \
--top_p 0.95 \
```

--max_new_tokens 512

Evaluation

We use the ms-swift inference framework. Dataset organization can follow eval_example.jsonl.

```bash
NPROC_PER_NODE=8 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=802816 \
swift eval \
--model /data/checkpoint-2161-merged \
--eval_backend Native \
--infer_backend pt \
--device_map auto \
--eval_limit 300 \
--eval_dataset general_qa \
```

--dataset_args '{"general_qa": {"local_path": "/root/Qwen2.5-VL/yanke/qa", "subset_list": ["eval_example"]}}'

## 🤝 Acknowledgements

This work was supported by the “ ” ( ) and the “ ” ( ).
