# CF-VLM : CounterFactual Vision-Language Fine-tuning
> Official code for the NIPS 2025 paper [“CF-VLM : CounterFactual Vision-Language Fine-tuning”](https://arxiv.org/abs/2506.17267).

\[ [English](README.md) | [中文](README_zh.md) \]

![CF-VLM Logo](https://img.shields.io/badge/NIPS-2025-blue) ![Python](https://img.shields.io/badge/Python-3.9%2B-green)

---

## Abstract

Recent advances in vision-language models (VLMs) have greatly improved crossmodal semantic understanding, yet significant limitations remain in fine-grained
discrimination and deep causal reasoning tasks. Existing VLMs often rely on
superficial statistical correlations, lacking the ability to capture the underlying
causal logic between visual and textual content. To address this, we propose
CounterFactual Vision-Language Fine-tuning (CF-VLM), a novel framework that
enhances the causal reasoning capabilities of VLMs through the targeted use of
counterfactual samples. CF-VLM introduces three complementary training objectives: maintaining foundational cross-modal alignment, reinforcing the uniqueness,
and stability of factual scene representations against coherent counterfactuals,
and sharpening the model’s sensitivity to minimal but critical causal edits. Extensive experiments demonstrate that CF-VLM consistently outperforms strong
baselines and state-of-the-art methods on compositional reasoning and generalization benchmarks. Furthermore, it shows promise in mitigating visual hallucinations,
indicating improved factual consistency. Our CF-VLM provides a robust foundation for deploying VLMs in high-stakes, real-world scenarios requiring reliable
reasoning and interpretability.


See the [paper](https://arxiv.org/abs/2506.17267) for theoretical details and full experiments.
---


## Citation

If you use this project, please cite:
```bibtex
@misc{zhang2025cfvlmcounterfactualvisionlanguagefinetuning,
      title={CF-VLM:CounterFactual Vision-Language Fine-tuning}, 
      author={Jusheng Zhang and Kaitong Cai and Yijia Fan and Jian Wang and Keze Wang},
      year={2025},
      eprint={2506.17267},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.17267}, 
}
```

---
