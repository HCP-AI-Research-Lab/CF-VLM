# CF-VLM : 反事实视觉-语言微调
> NIPS 2025 论文 [“CF-VLM : CounterFactual Vision-Language Fine-tuning”](https://arxiv.org/abs/2506.17267) 的官方代码。  
<div align="center">
<img src="Mainflow.png" width="1000">
</div>

[English](README.md) | [中文](README_zh.md)

![CF-VLM Logo](https://img.shields.io/badge/NIPS-2025-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)

---

## 摘要

近年来，视觉-语言模型（VLMs）在跨模态语义理解方面取得了显著进展，但在细粒度判别与深层因果推理任务上仍存在明显局限。现有 VLM 常依赖表层统计相关，难以刻画视觉与文本之间的因果逻辑。为此，我们提出 **CounterFactual Vision-Language Fine-tuning (CF-VLM)**：通过**有针对性地引入反事实样本**，在不破坏基础跨模态对齐的前提下，增强模型在**唯一性/稳定性**与**关键因果微编辑**上的敏感度，从而提升组合推理、泛化与事实一致性。大量实验显示，CF-VLM 在多项推理基准上优于强基线与最新方法，并对减轻视觉幻觉具有潜在帮助。

详见[论文](https://arxiv.org/abs/2506.17267)获取理论细节与完整实验。

---

## 目录
- [安装](#安装)
- [环境要求](#环境要求)
- [数据准备](#数据准备)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [训练](#训练)
- [推理与评测](#推理与评测)
- [常见问题](#常见问题)
- [许可协议](#许可协议)
- [引用](#引用)
- [致谢](#致谢)
- [联系](#联系)

---

## 安装

1. 克隆仓库
   ```bash
   git clone https://github.com/your_org/CF-VLM.git
   cd CF-VLM
   ```

2. （可选）创建虚拟环境
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # Linux / macOS
   # .\.venv\Scripts\activate  # Windows PowerShell
   ```

3. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

4. 配置 Qwen2.5-VL 推理模型  
   请参照官方文档完成部署与权限配置：  
   👉 Qwen 官方文档：https://github.com/QwenLM/Qwen2.5-VL  

---

## 环境要求

- Python **3.9+**
- PyTorch **2.1+**
- CUDA **11.8+**
- NVIDIA GPU (A100/80GB 推荐)
- 依赖包见 `requirements.txt`

---

## 数据准备

运行 `process.py` 生成反事实数据：
```bash
python process.py --input_path data/raw --output_path data/counterfactual --num_workers 8 --seed 42
```

---

## 目录结构

```
CF-VLM/
├─ process.py
├─ clip_train.py
├─ Qwen_train.py
├─ requirements.txt
├─ README.md
└─ README_zh.md
```

---

## 快速开始

1. 生成反事实数据
   ```bash
   python process.py
   ```

2. 训练 CLIP 模型
   ```bash
   python clip_train.py
   ```

3. 训练 Qwen 模型
   ```bash
   python Qwen_train.py
   ```

---

## 引用

如果您觉得本项目有帮助，请引用：
```bibtex
@article{cfvlm2025,
  title={CF-VLM: CounterFactual Vision-Language Fine-tuning},
  author={Your Name et al.},
  journal={NeurIPS},
  year={2025}
}
```
