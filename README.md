# 🎨 AI-Powered Fashion Article Image Generation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Transformers](https://img.shields.io/badge/Transformers-4.40%2B-yellow)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2F12.x-green)
![License](https://img.shields.io/badge/License-MIT-brightgreen)
![Status](https://img.shields.io/badge/Status-Production-success)

---

## 📋 Overview

This project implements a **cutting-edge multi-model AI pipeline** that automatically generates **photorealistic fashion product images** from German text descriptions. Built for **NKD**, a leading German fashion retailer, this system eliminates the need for costly and time-consuming photography shoots by leveraging state-of-the-art **Large Language Models (LLMs)** and **Diffusion-based Image Generation**.

The pipeline transforms simple textual product descriptions into high-quality, catalog-ready images suitable for e-commerce, marketing materials, and digital catalogs—all without human photographers, models, or physical samples.

### **The Problem:**
Traditional fashion catalog creation requires:
- Expensive photography equipment and studios
- Professional photographers and models
- Physical product samples
- Time-consuming post-processing
- High operational costs (€50-200 per product shoot)

### **The Solution:**
A fully automated AI pipeline that:
- Generates photorealistic images in **42 seconds** per product
- Achieves **87.6% first-pass success rate**
- Produces **85%+ catalog-ready images** without manual intervention
- Reduces costs by **~90%** compared to traditional photography
- Scales effortlessly from 1 to 10,000+ products

---

## ✨ Key Features

### **🤖 Multi-Model AI Architecture**
- **4 Neural Networks in Sequence**: Translation → LLM Extraction → Diffusion → CLIP Scoring
- **End-to-End Automation**: From German text to final image with zero manual intervention
- **Production-Grade Pipeline**: Handles batch processing of entire product catalogs

### **🌐 German-to-English Neural Translation**
- Automated language conversion using Helsinki-NLP's MarianMT model
- Preserves fashion-specific terminology and product attributes
- Enables downstream models trained on English corpora

### **🧠 LLM-Based Structured Attribute Extraction**
- Microsoft Phi-3 Mini (3.8B parameters) extracts structured JSON from unstructured text
- **94% attribute retention accuracy**
- Robust error handling with regex-based fallback parsing
- Extracts: product type, visual features, colors, materials, design aspects

### **🎨 Advanced Diffusion-Based Image Synthesis**
- **FLUX.1-schnell**: Fast, high-quality diffusion model optimized for inference speed
- **Fashion LoRA Fine-Tuning**: Domain-specific adaptation for clothing, fabrics, and accessories
- **42 seconds average generation time** per image
- Photorealistic textures, accurate colors, and professional studio lighting

### **📊 Automated Quality Assessment with CLIP**
- OpenAI CLIP (ViT-Large) measures image-text semantic alignment
- Zero-shot quality scoring without human annotation
- **Correlation**: CLIP scores >30 → 87% passing rate in human evaluation

### **🔄 Reproducibility & Determinism**
- Complete seed control across all stochastic components
- Deterministic generation for A/B testing and experimentation
- Essential for production ML systems

### **📦 Batch Processing & Metadata Tracking**
- Processes entire Excel datasets automatically
- Generates JSON metadata for each image (prompts, scores, paths)
- Statistical summaries (mean, median, std dev of CLIP scores)

---

## 🛠️ Technologies Used

### **Deep Learning Frameworks**
- **PyTorch** (2.0+) - Core deep learning framework with CUDA acceleration
- **Hugging Face Transformers** (≥4.40.0) - LLM and translation models
- **Hugging Face Diffusers** (≥0.27.0) - Diffusion model inference
- **Accelerate** - Multi-GPU model parallelism

### **AI Models**
1. **Helsinki-NLP/opus-mt-de-en** - Neural Machine Translation (MarianMT architecture)
2. **Microsoft Phi-3 Mini** (phi-3-mini-4k-instruct) - Small Language Model for attribute extraction
3. **FLUX.1-schnell** + **aihpi/flux-fashion-lora** - Fast diffusion model with fashion-specific fine-tuning
4. **OpenAI CLIP** (clip-vit-large-patch14) - Vision-language model for quality assessment

### **Data Processing & Utilities**
- **pandas** - Excel file handling and data manipulation
- **openpyxl** - Excel parsing (.xlsx files)
- **numpy** - Numerical operations and array handling
- **Pillow (PIL)** - Image manipulation and saving
- **json, re (regex)** - Data parsing and prompt cleaning

### **Additional Libraries**
- **statistics** - Performance metric calculations
- **SentencePiece** - Tokenization for translation models
- **torch.Generator** - Reproducible random number generation

---

## 📁 Folder Structure

```
fashion-article-image-generation/
│
├── Article_img_generation.py          # Main pipeline script (4-model integration)
├── Execution_Flow.txt                 # Detailed architecture documentation
├── Requirements.txt                   # Installation guide and system requirements
├── Article_image_generation.pdf       # Complete research report with evaluation
│
├── Dataset/
│   ├── article_descriptions_1.xlsx    # German product descriptions (batch 1)
│   ├── article_descriptions_2.xlsx    # German product descriptions (batch 2)
│   └── article_descriptions_3.xlsx    # German product descriptions (batch 3)
│
├── output/                            # Generated images and metadata
│   ├── 0.png                          # Generated product image (sample 0)
│   ├── 0.json                         # Metadata: prompts, CLIP score, paths
│   ├── 1.png
│   ├── 1.json
│   ├── ...
│   └── clip_statistics.txt            # Performance summary (mean, std, min, max)
│
└── sample_output/                     # Showcase examples for README/presentation
    ├── 6.png                          # Children's girl's t-shirt with embroidery
    ├── 6.json
    ├── 17.png                         # Boys' dinosaur print t-shirt
    ├── 17.json
    ├── 70.png                         # Additional showcase example
    └── grid_output.jpg                # 4-image grid showcasing variety
```

---

## 🚀 Getting Started

### **Prerequisites**

#### **Hardware Requirements:**
- **Operating System**: Linux (Ubuntu 20.04+ recommended) or Windows with WSL2
- **GPU**: NVIDIA GPU with **≥12GB VRAM** (RTX 3060, RTX 3080, A4000, or better)
- **CUDA**: Version 11.8 or 12.x (must match PyTorch installation)
- **RAM**: 16GB+ system RAM recommended
- **Storage**: 50GB+ free space for models and cache

#### **Software Requirements:**
- **Python**: 3.9 or 3.10 (verified compatibility)
- **Hugging Face Account**: Required for model downloads
- **Hugging Face Token**: Create at https://huggingface.co/settings/tokens

---

### **Installation**

#### **Step 1: Clone the Repository**
```bash
git clone https://github.com/yourusername/fashion-article-image-generation.git
cd fashion-article-image-generation
```

#### **Step 2: Create Virtual Environment (Recommended)**
```bash
# Using conda (recommended)
conda create -n fashion-ai python=3.10
conda activate fashion-ai

# OR using venv
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

#### **Step 3: Install PyTorch with CUDA Support**
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### **Step 4: Install Core Dependencies**
```bash
pip install transformers>=4.40.0 diffusers>=0.27.0 accelerate sentencepiece
pip install pandas openpyxl pillow numpy scipy
```

#### **Step 5: Hugging Face Authentication**
```bash
huggingface-cli login
# Enter your Hugging Face token when prompted
```

#### **Step 6: Download Required Models**
```bash
# LLM and Translation Models
huggingface-cli download microsoft/phi-3-mini-4k-instruct
huggingface-cli download Helsinki-NLP/opus-mt-de-en

# CLIP for Quality Scoring
huggingface-cli download openai/clip-vit-large-patch14

# Diffusion Pipeline
huggingface-cli download black-forest-labs/FLUX.1-schnell
huggingface-cli download aihpi/flux-fashion-lora --include="*.safetensors"
```

#### **Step 7: Set Environment Variables**
```bash
# Set cache directory (optional, to save space)
export HF_HOME="/path/to/your/cache"
export HF_DATASETS_CACHE="/path/to/your/cache"

# OR add to your script
# os.environ["HF_HOME"] = "/path/to/your/cache"
```

---

## 💻 Usage

### **Basic Usage**

Run the main pipeline script to process your product descriptions:

```bash
python Article_img_generation.py
```

### **What the Script Does:**

1. **Loads Product Descriptions**: Reads Excel files from `Dataset/` folder
2. **Translates Text**: German → English using Helsinki-NLP translator
3. **Extracts Attributes**: Structured JSON extraction via Phi-3 Mini LLM
4. **Generates Prompts**: Creates detailed prompts for image generation
5. **Synthesizes Images**: Uses FLUX.1-schnell diffusion model with fashion LoRA
6. **Calculates Quality Scores**: CLIP-based semantic alignment scoring
7. **Saves Outputs**: Images (.png) + Metadata (.json) in `output/` directory
8. **Generates Statistics**: CLIP score summary saved to `clip_statistics.txt`

### **Expected Runtime:**

- **Single Image**: ~42 seconds
- **224 Articles**: ~2.5 hours (full dataset)
- **Processing Speed**: ~85 images/hour

### **Output Format:**

Each processed article generates:

**Image File** (`output/N.png`):
- High-resolution photorealistic product image
- Clean white background, studio lighting
- Perfectly centered, flat lay perspective

**Metadata File** (`output/N.json`):
```json
{
  "sample_id": 6,
  "original_description": "Kinder-Mädchen-T-Shirt mit Stickerei...",
  "generation_prompt": "Professional product photography of a Children's girl's T-shirt...",
  "clip_prompt": "Product image of a Children's girl's T-shirt",
  "clip_score": 28.31,
  "image_path": "output/6.png"
}
```

**Statistics Summary** (`output/clip_statistics.txt`):
```
CLIP Score Statistics:
Total Samples: 224
Mean: 32.45
Median: 31.78
Std Dev: 4.23
Min: 24.17
Max: 41.62
```

---

## 📊 Results

### **Performance Metrics**

| Metric | Value |
|--------|-------|
| **First-Pass Success Rate** | 87.6% |
| **Images Passing Evaluation** | 85%+ |
| **Category Accuracy** | 89% |
| **Color Fidelity** | 92% |
| **Design Element Accuracy** | 95% |
| **Attribute Retention (Prompts)** | 94% |
| **Average Generation Time** | 42 seconds/image |
| **Dataset Processing Time** | 2.5 hours (224 articles) |
| **Visual Artifact Rate** | 3% |

---

### **Generated Image Examples**

#### **Sample 6: Children's Girl's T-Shirt with Embroidery**

![Sample 6](sample_output/6.png)

**Original German Description:**
> "Kinder-Mädchen-T-Shirt mit Stickerei. Die bestickten Ärmel und die Stickerei 'Dream Journey' verleihen dem Shirt eine besondere Note."

**Generated Prompt:**
> "Professional product photography of a Children's girl's T-shirt, featuring embroidered sleeves, embroidery 'Dream Journey', high cotton content, OEKO-TEX® Standard 100 certified, placed on a clean white background, laid flat, studio lighting, 8k, 85mm lens f/2.8, no humans, no mannequins, perfectly centered"

**CLIP Score:** 28.31

**Analysis:**
- ✅ **Perfect "Dream Journey" text rendering** in golden embroidery
- ✅ **Colorful floral sleeve embroidery** (pink, red, yellow, turquoise flowers)
- ✅ **Black fabric texture** accurately rendered
- ✅ **Studio-quality lighting** with subtle shadows
- ✅ **Clean white background**, perfectly centered composition

---

#### **Sample 17: Boys' Dinosaur Print T-Shirt**

![Sample 17](sample_output/17.png)

**Original German Description:**
> "Jungen-T-Shirt mit Dino-Frontaufdruck. Der schicke Dino-Frontaufdruck mit verschiedenen Dinosauriern begeistert kleine Urzeit-Fans. Die angesagte Melange-Optik macht den modernen Look perfekt."

**Generated Prompt:**
> "Professional product photography of a Boys' T-shirt, featuring Dino front print, various dinosaurs, trendy melange look, soft material, high cotton content, placed on a clean white background, laid flat, studio lighting, 8k, 85mm lens f/2.8, no humans, no mannequins, perfectly centered"

**CLIP Score:** 27.85

**Analysis:**
- ✅ **Multiple colorful dinosaurs** rendered in detail (blue, green, orange, brown)
- ✅ **Grey melange fabric texture** accurately captured
- ✅ **All-over print pattern** with excellent color saturation
- ✅ **Professional catalog-ready quality**
- ✅ **Consistent white background** and centered framing

---

#### **Showcase Grid: Multi-Category Generation**

![Grid Output](sample_output/grid_output.jpg)

**4-Image Showcase Demonstrating Pipeline Versatility:**

1. **White Blouse with Front Knot** (Women's)
   - Elegant knotted design at waist
   - Clean white fabric, subtle texture
   - Professional studio presentation

2. **Olive Green Men's Shirt** (Men's)
   - Button-down with chest pocket
   - Accurate color rendering (olive/khaki)
   - Kent collar, short sleeves

3. **Black Mickey Mouse T-Shirt** (Women's)
   - Iconic character print in red/white
   - Crew neck, comfortable fit
   - Licensed character accurately rendered

4. **Colorful Dinosaur Pattern T-Shirt** (Children's)
   - Vibrant multi-color dinosaur print
   - Grey melange base fabric
   - Playful, child-appropriate design

**Key Achievements:**
- ✅ **Cross-Category Consistency**: Maintains quality across men's, women's, and children's wear
- ✅ **Complex Pattern Rendering**: Handles characters, prints, and embroidery
- ✅ **Fabric Texture Variety**: Differentiates between cotton, melange, and blended materials
- ✅ **Color Accuracy**: From white to black to vibrant multi-color prints

---

### **Quality Assessment Framework**

The project uses a **custom deductive scoring system (0-10 scale)** developed for fashion e-commerce:

#### **Evaluation Criteria:**

| Criterion | Weight | Deduction |
|-----------|--------|-----------|
| **Product Type Match** | Critical | -1.5 (strong deviation) |
| **Color Accuracy** | High | -0.5 to -1.5 |
| **Design Elements** | High | -1.5 (missing key features) |
| **Fabric Texture** | Medium | -0.5 (minor mismatch) |
| **Background Consistency** | Low | -0.5 (slight issues) |
| **Lighting Quality** | Low | -0.5 (unnatural shadows) |

#### **Scoring Thresholds:**

- **9.5-10.0**: Pass - Excellent (production-ready, no modifications needed)
- **8.5-9.5**: Pass (minor issues, visually acceptable)
- **<8.5**: Fail (requires regeneration or manual correction)

#### **Results Distribution:**

- **85%+ images passed** evaluation (scores ≥8.5)
- **60%+ achieved near-perfect scores** (9.5-10.0)
- **Average score**: 9.2/10

---

### **Model Performance Comparison**

During development, multiple diffusion models were evaluated:

| Model | Image Quality | Speed | Artifacts | Prompt Adherence | Selected? |
|-------|---------------|-------|-----------|------------------|-----------|
| **OpenAI DALL-E** | Excellent | Slow | Minimal | High (overfitting risk) | ❌ |
| **Stable Diffusion 1.5** | Fair | Fast | Moderate | Low | ❌ |
| **Stable Diffusion 2.1** | Fair | Fast | High (unwanted objects) | Low | ❌ |
| **SD 1.5 + LoRA** | Good | Fast | Low | Medium (stylized) | ❌ |
| **DL Photoreal** | Good | Medium | High (human figures) | Medium | ❌ |
| **FLUX.1-dev** | Excellent | Medium | Minimal | Excellent | ⚠️ (20GB VRAM) |
| **FLUX.1-schnell** | Excellent | **Fast** | Minimal | Excellent | ✅ **Selected** |

**Winner:** **FLUX.1-schnell** - Optimal balance of quality, speed, and resource efficiency.

---

### **Technical Implementation Highlights**

#### **1. Reproducibility Through Comprehensive Seeding**
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
**Why it matters:**
- Ensures identical outputs across multiple runs
- Critical for A/B testing prompt strategies
- Enables scientific experimentation with controlled variables

#### **2. LoRA Fine-Tuning for Fashion Domain**
```python
pipe.load_lora_weights("aihpi/flux-fashion-lora")
```
**Benefits:**
- **Parameter-Efficient**: <1% parameters vs full fine-tuning
- **Fashion-Specific**: Better fabric textures, garment structure, and draping
- **Preserves Base Model**: Retains general photorealism while adding domain expertise

#### **3. Two-Stage Prompt Engineering**

**Simple Prompt (for CLIP scoring):**
```
"Product image of a Children's girl's T-shirt"
```

**Detailed Prompt (for image generation):**
```
"Professional product photography of a Children's girl's T-shirt, 
featuring embroidered sleeves, embroidery 'Dream Journey', 
high cotton content, placed on a clean white background, laid flat,
studio lighting, 8k, 85mm lens f/2.8, no humans, no mannequins, 
no zoom, no tilt, no rotation, perfectly centered"
```

**Rationale:**
- Detailed prompts guide diffusion model's visual attributes
- Simple prompts avoid over-specification in CLIP evaluation
- Photography terminology (lens, lighting) conditions artistic style

---

## 🔮 Future Improvements

Based on the current implementation and evaluation results, here are potential enhancements:

### **1. Multi-Variant Image Generation**

**Current Limitation**: Single front-view images only.

**Proposed Enhancement**:
- Generate **multiple angles** (front, back, side, close-up details)
- Implement **ControlNet conditioning** for pose/angle control
- Create **lifestyle images** (on models, styled environments)

**Impact**:
- Complete 360° product catalog coverage
- Enhanced customer experience and visualization
- Reduced need for multiple photography sessions

**Technical Approach**:
```python
angles = ["front view", "back view", "side view", "detail close-up"]
for angle in angles:
    prompt = f"{base_prompt}, {angle}, studio lighting..."
    image = pipe(prompt, generator=g).images[0]
```

---

### **2. LLM Fine-Tuning for Fashion Domain**

**Current Challenge**: Generic Phi-3 Mini occasionally oversimplifies fashion terminology.

**Proposed Solution**:
- **Fine-tune Phi-3 Mini** on fashion-specific corpus (1,000+ examples)
- Implement **LoRA-based parameter-efficient fine-tuning**
- Add **glossary-based RAG** for technical terms (e.g., "A-line", "French terry")

**Impact**:
- Attribute extraction accuracy: **94% → 98%+**
- Better handling of specialized fashion vocabulary
- Reduced manual correction needs

**Training Requirements**:
- Dataset: 1,000+ German descriptions with labeled JSON
- Training time: 2-4 hours on single GPU
- Cost: Minimal (use existing hardware)

---

### **3. Interactive Web Interface (Gradio/Streamlit)**

**Current Limitation**: Batch processing only, no real-time user control.

**Proposed Solution**:
- Build **web-based UI** for real-time generation
- Allow **manual prompt editing** before image generation
- Implement **iterative refinement** workflow

**Impact**:
- Empowers non-technical users (designers, marketing teams)
- Faster iteration for challenging edge cases
- Creative control for special collections

**UI Features**:
- Text input for German descriptions
- Editable JSON attribute fields
- Live image preview with CLIP scores
- Download/regenerate buttons

---

## 📄 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Areas for Contribution:**
- Multi-angle image generation with ControlNet
- UI/UX development for web interface
- Performance optimizations (quantization, TensorRT)
- Multi-language support (French, Spanish descriptions)

---

## 👨‍💻 Author

**Nayeemuddin Mohammed**  
Master's Student - Applied AI for Digital Production Management  
Deggendorf Institute of Technology, Germany

- GitHub: [@thelostbong](https://github.com/thelostbong)
- LinkedIn: [Nayeemuddin-Mohammed-03](https://linkedin.com/in/nayeemuddin-mohammed-03/)
- Email: nayeemuddin.mohammed@th-deg.de

---

## 🙏 Acknowledgments

### **Academic Supervision:**
- **Dr.-Ing. Sunil P. Survaiya** - Professor, Deggendorf Institute of Technology
  - Guidance on AI pipeline architecture and evaluation methodology

### **Industry Partnership:**
- **NKD (German Fashion Retail)**
  - **Dr. Johannes Schöck** - Technical Collaboration
  - **Florian K.T. Scheibner** - Product Description Dataset

### **Open-Source Community:**
- **Hugging Face** - Transformers, Diffusers libraries, model hosting
- **Black Forest Labs** - FLUX.1 diffusion model
- **Microsoft Research** - Phi-3 Mini LLM
- **OpenAI** - CLIP vision-language model
- **Helsinki-NLP** - Opus-MT translation models

### **Infrastructure:**
- **Deggendorf Institute of Technology** - GPU server access

---

## 📚 References

1. Rombach, R., et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR 2022.
2. Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021.
3. Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
4. Abdin, M., et al. (2024). *Phi-3 Technical Report*. Microsoft Research.

---

<div align="center">

### **⭐ If you found this project helpful, please consider giving it a star! ⭐**

**Built with 🤖 AI, ❤️ Passion, and ☕ Coffee at Deggendorf Institute of Technology**

</div>
