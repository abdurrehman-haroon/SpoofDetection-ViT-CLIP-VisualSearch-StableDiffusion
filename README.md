# SpoofDetection-ViT-CLIP-VisualSearch-StableDiffusion

### *Computer‑Vision Mini‑Suite: Anti‑Spoofing • Visual Search • Image Variation*

---

<div align="center">
  <b>Author &nbsp;:</b> Abdurrehman Haroon  •  <b>Course :</b> L21‑5691 Generative AI  •  <b>License :</b> MIT  
  <b>Last update:</b> May 2025
</div>

---


## Project Overview

This repository bundles three independent yet complementary computer‑vision demos:

| Module                             | Goal                                                             | Backbone              | Dataset                        |
| ---------------------------------- | ---------------------------------------------------------------- | --------------------- | ------------------------------ |
| **1. Spoof Detection**             | Spot printed‑photo or replay attacks                             | ViT‑Base (timm)       | *nguyenkhoa/celeba‑spoof* (HF) |
| **2. Visual Search**               | Retrieve the 5 most semantically similar images for a text query | CLIP ViT‑B/32         | COCO 2017 (val)                |
| **3. Stable Diffusion Variations** | Generate creative variants of a given image                      | Stable Diffusion v1.5 | Any user image                 |

All experiments were run on an **NVIDIA RTX 3050 Laptop GPU** using PyTorch 2 + HF Transformers & Diffusers.

---

## Key Components

* **Pure PyTorch notebooks** – end‑to‑end code inside `notebooks/`.
* **Lightning‑fast ViT spoofing** – \~93 % accuracy on CelebA‑Spoof.
* **Zero‑shot CLIP retrieval** – cosine similarity on frozen embeddings.
* **Plug‑and‑play Stable Diffusion** – control `strength`, `guidance_scale`, `num_inference_steps` via CLI.

---

## Getting Started

### Prerequisites

```bash
Python >=3.9
pip install torch torchvision timm transformers datasets diffusers accelerate opencv-python matplotlib
```

A GPU is strongly recommended for Stable Diffusion.

### Installation

```bash
git clone https://github.com/abdurrehman-haroon/SpoofDetection-ViT-CLIP-VisualSearch-StableDiffusion
cd SpoofDetection-ViT-CLIP-VisualSearch-StableDiffusion
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Spoof Detection (ViT)

**Pipeline**

1. **Dataset split** – 20 % train / 80 % test from CelebA‑Spoof.
2. **Pre‑processing** – Hugging Face `AutoFeatureExtractor` → 224×224 RGB tensors.
3. **Model** – `timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)`.
4. **Training** – Adam 1e‑4, batch = 8, 1 epoch (demo) → extend for real use.
5. **Metrics** – Accuracy, Precision, Recall, F1.

> **Result** – *Accuracy ≈ 93 %, F1 ≈ 0.93*

False‑positives mainly stem from high‑resolution printed photos; improve via augmentation & extra epochs.

---

## AI‑Powered Visual Search (CLIP)

**Workflow**

1. Encode **5000 COCO‑val images** → store normalized embeddings.
2. Encode text query ➜ cosine similarity against image embeddings.
3. Return **top k** images + best matching COCO caption.

Sample query *"a cat in a house"* retrieved five relevant images with image‑sim ≈ 0.30 and caption‑sim ≈ 0.84 citeturn1file1.

---

## Stable Diffusion Image Variation

Uses Hugging Face **Diffusers** `StableDiffusionPipeline` in *image‑to‑image* mode.

| Parameter             | Description                     | Typical Range |
| --------------------- | ------------------------------- | ------------- |
| `strength`            | How much to transform the input | 0.3 – 0.8     |
| `guidance_scale`      | Classifier‑free guidance weight | 7 – 12        |
| `num_inference_steps` | Diffusion steps                 | 25 – 100      |

Generated examples (see `outputs/variations/`) illustrate style transfer and content preservation.

---

## Results & Discussion

* **Spoof Detection** – Robust with limited compute; recall > precision → safe for security use‑cases.
* **Visual Search** – CLIP embeddings capture scene semantics; occasional background bias.
* **Stable Diffusion** – High‑fidelity variations; stronger guidance = closer to prompt.

---

## Troubleshooting & FAQ

| Issue                           | Fix                                                            |
| ------------------------------- | -------------------------------------------------------------- |
| CUDA out‑of‑memory              | Reduce batch, use `torch.cuda.set_per_process_memory_fraction` |
| Wrong spoof label on real image | Add real‑world augmentations (blur, noise)                     |
| CLIP retrieval slow             | Cache embeddings to disk (`pickle`)                            |
| SD outputs washed‑out           | Lower `strength`, increase `steps`                             |

---

## Contributing

Pull requests and suggestions are welcome!  Please follow the **PEP‑8** style guide and include unit tests.

---

## License

Released under the MIT License.  See `LICENSE` for full text.

---


