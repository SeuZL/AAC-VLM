# 👁️ AAC-VLM: Attribute-Anchored Collaborative Vision-Language Framework for Multi-Image Clinical Report Generation

AAC-VLM is an attribute-anchored collaborative vision-language framework for multi-image clinical report generation. It is designed for clinical workflows in which multiple medical images support a single integrated report, such as fundus fluorescein angiography (FFA).

AAC-VLM introduces an Evidence Anchoring Auxiliary module (EAA) as a visual-semantic anchor. The framework reconstructs image-relevant training supervision through sentence-level filtration, separates evidence-centered description from structured report generation through a two-stage prompting strategy, and aggregates frame-level outputs into an examination-level report.

AAC-VLM is intended as a clinician-supervised structured drafting tool. Final interpretation and report sign-off must be performed by clinicians.

---

## 📑 Table of Contents

* [Key Highlights](#-key-highlights)
* [Framework Overview](#️-framework-overview)
* [Environment Setup](#️-environment-setup)
* [Data Preparation](#-data-preparation)
* [Base Models](#-base-models)
* [Pipeline](#-pipeline)
* [Reproducibility Notes](#-reproducibility-notes)
* [Acknowledgements](#-acknowledgements)

---

## 🌟 Key Highlights

* **Evidence-anchored supervision reconstruction:** EAA predicts image-relevant clinical terms and supports sentence-level filtration of integrated reports, reducing mismatch between individual frames and examination-level report text.

* **Two-stage structured generation:** AAC-VLM first generates an evidence-centered free-text description and then converts it into a structured report under clinical formatting requirements.

* **Cross-frame aggregation:** Frame-level structured reports are deduplicated and merged into an examination-level output while clinically meaningful inter-frame differences are retained.

* **Clinical evaluation:** In blinded clinician review, 70.18% of generated reports were rated clinically usable as editable drafts.

---

## 🏗️ Framework Overview

AAC-VLM contains three core components:

1. **Evidence Anchoring Auxiliary module (EAA)**
   Extracts image-relevant clinical terms from complete FFA frames.

2. **Evidence-anchored sentence-level filtration**
   Retains report sentences whose clinical terms match the current image and removes unsupported descriptions.

3. **Two-stage structured prompting and cross-frame aggregation**
   Generates an evidence-centered free-text description, converts it into a structured single-image report, and aggregates frame-level outputs into an examination-level report.

---

## 🛠️ Environment Setup

This project was developed on Ascend 910 NPUs and uses the PyTorch ecosystem.

### Clone the repository

```bash
git clone https://github.com/SeuZL/AAC-VLM.git
cd AAC-VLM
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

---

## 📂 Data Preparation

The internal clinical dataset is not publicly released because of patient privacy constraints, ethics approval requirements, and institutional data-governance regulations.

Public retinal image-text datasets used for external testing can be obtained from their original sources:

* **FFA-IR:** https://physionet.org/content/ffa-ir-medical-report/1.1.0/
* **MM-Retinal v1:** https://github.com/lxirich/MM-Retinal

A suggested directory structure is:

```text
data/
├── ffa_ir/
│   ├── images/
│   └── annotations.jsonl
├── mm_retinal_v1/
│   ├── images/
│   └── annotations.jsonl
└── custom_data/
    ├── images/
    └── labels.jsonl
```

The training data follow a multi-image, single-report setting. To construct image-level supervision, each integrated report is first paired with the corresponding images and converted into JSONL format. EAA predictions are then used for sentence-level filtration.

### Example JSONL item

```json
{
  "messages": [
    {
      "role": "user",
      "content": "You are an ophthalmologist proficient in fundus diseases. Carefully read the image and write a textual description.",
      "image": "data/custom_data/images/example.png"
    },
    {
      "role": "assistant",
      "content": "Macular vessels are tortuous; a WISS ring is visible superior to the optic disc."
    }
  ]
}
```

---

## 📦 Base Models

### Evidence Anchoring Auxiliary module

* **Backbone:** EfficientNet-B0
* **Reference implementation:** https://huggingface.co/timm/efficientnet_b0.ra_in1k

### Vision-language backbone

* **Backbone:** Qwen2.5-VL-32B-Instruct
* **Model page:** https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct

---

## 🚀 Pipeline

### Step 1. Train the EAA

Prepare the multilabel classification dataset following the format in `multi_label_data.jsonl`.

```bash
python trainclassify.py
```

The EAA is used as an auxiliary multilabel attribute-recognition module. The experimental configuration reported in the manuscript uses EfficientNet-B0, binary cross-entropy loss, Adam optimization, an initial learning rate of `1e-4`, a batch size of `16`, and a probability threshold of `0.5`.

---

### Step 2. Predict image-level clinical terms

Edit the input, output, and checkpoint paths in `fortagdata.py`, then run:

```bash
python fortagdata.py
```

The script appends EAA-predicted clinical terms to each image-level JSONL item.

### Example output

```json
{
  "messages": [
    {
      "role": "user",
      "content": "You are an ophthalmologist proficient in fundus diseases. Carefully read the image and write a textual description.",
      "image": "data/custom_data/images/example.png"
    },
    {
      "role": "assistant",
      "content": "Macular vessels are tortuous; a WISS ring is visible superior to the optic disc."
    }
  ],
  "tag": [
    "Macula",
    "Fovea",
    "Optic disc",
    "Artery",
    "Vein",
    "Capillary",
    "Posterior pole",
    "Perifoveal ring"
  ]
}
```

---

### Step 3. Perform sentence-level filtration

Edit the file paths in `selecttag.py`, then run:

```bash
python selecttag.py
```

The resulting JSONL file contains image-relevant textual supervision for VLM fine-tuning.

---

### Step 4. Fine-tune the vision-language backbone

The following command illustrates the LoRA fine-tuning configuration used for the Qwen2.5-VL-32B backbone.

Replace the paths with your local model, dataset, DeepSpeed configuration, and output directory.

```bash
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_CHECK_TIMEOUT=600
export MAX_PIXELS=802816
export TORCH_NPU_FUSION_ENABLE=1

torchrun \
  --nproc_per_node=8 \
  --rdzv_conf "overlap_timeout=600" \
  /path/to/ms-swift/swift/cli/sft.py \
  --ddp_backend hccl \
  --model /path/to/Qwen2.5-VL-32B-Instruct \
  --dataset /path/to/filtered_training_data.jsonl \
  --deepspeed /path/to/deepspeed_zero3.json \
  --train_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.1 \
  --use_rslora false \
  --max_grad_norm 1.0 \
  --target_modules all-linear \
  --freeze_vit true \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 100 \
  --logging_steps 50 \
  --max_length 8192 \
  --output_dir /path/to/output \
  --dataloader_num_workers 8 \
  --model_kwargs '{"device_map":{"":"npu:auto"}}' \
  --optim adamw_hf \
  --gradient_checkpointing true
```

---

### Step 5. Stage-1 inference: evidence-centered description

Prepare a JSONL dataset using the Stage-1 prompt.

### Stage-1 prompt example

```json
{
  "messages": [
    {
      "role": "user",
      "content": "You are an ophthalmologist proficient in fundus diseases. You are given a right-eye FFA image. Patient: female, 51 years old. The model predicts that the image may contain the following tags: Macula, Fovea, Optic disc, Artery, Vein, Capillary, Posterior pole, Perifoveal ring. Carefully read the image and write an interpretation report.",
      "image": "data/custom_data/images/example.png"
    }
  ]
}
```

Run inference:

```bash
NPROC_PER_NODE=4 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=802816 \
swift infer \
  --adapters /path/to/checkpoint \
  --val_dataset /path/to/stage1_input.jsonl \
  --infer_backend pt \
  --max_batch_size 1 \
  --device_map auto \
  --temperature 0.1 \
  --repetition_penalty 1 \
  --top_p 0.9 \
  --max_new_tokens 512
```

---

### Step 6. Stage-2 inference: structured report generation

Prepare a second JSONL dataset by injecting the Stage-1 output and the structured reporting rules into the Stage-2 prompt.

### Stage-2 prompt template

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<FULL_STANDARDIZATION_RULES>\n\nOther information the image may contain:\n<STAGE1_OUTPUT>",
      "image": "data/custom_data/images/example.png"
    }
  ]
}
```

Run inference:

```bash
NPROC_PER_NODE=4 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=802816 \
swift infer \
  --adapters /path/to/checkpoint \
  --val_dataset /path/to/stage2_input.jsonl \
  --infer_backend pt \
  --max_batch_size 1 \
  --device_map auto \
  --temperature 0.1 \
  --repetition_penalty 1 \
  --top_p 0.9 \
  --max_new_tokens 512
```

---

### Step 7. Cross-frame aggregation

For each examination sequence, generate structured reports frame by frame. Deduplicate and merge report items by clinical field while retaining clinically meaningful inter-frame differences and inconsistencies.

The final output is an aggregated examination-level structured report.

---

### Step 8. Evaluation

Automatic evaluation in the manuscript uses:

* ROUGE-1 recall
* ROUGE-2 recall
* ROUGE-L recall
* BLEU-1

Additional evaluations include:

* ablation analysis
* hallucination-related failure-mode assessment
* blinded clinician review
* clinician-model workflow comparison
* AI-assisted reader study

Evaluation should be performed on the generated examination-level reports.

---

## 🔎 Reproducibility Notes

* The internal multicenter FFA dataset cannot be publicly released because of patient privacy constraints and institutional data-governance regulations.
* Public external datasets can be used for independent testing.
* Exact reproduction of internal-cohort results requires access to the restricted clinical dataset.
* The repository provides the core implementation workflow for data preparation, EAA training, sentence-level filtration, VLM fine-tuning, and two-stage inference.

---

## 🤝 Acknowledgements

This work was supported in part by the National Natural Science Foundation of China (grant no. 82571165), the Southeast University–Jiangsu Province Hospital Joint Open Organ Chip Project (grant no. 2024-K01), the Specialized Diseases Clinical Research Fund of Jiangsu Province Hospital (grant no. XB202404), and the Basic Research Program of Jiangsu (grant no. BK20243054). This work was partially supported by SEU Kunpeng & Ascend Center of Cultivation.
