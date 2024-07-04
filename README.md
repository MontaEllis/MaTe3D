

### <div align="center"> MaTe3D: Mask-guided Text-based 3D-aware Portrait Editing <div> 

[Kangneng Zhou](https://montaellis.github.io/), [Daiheng Gao](https://tomguluson92.github.io/), [Xuan Wang](https://xuanwangvc.github.io/), [Jie Zhang](https://scholar.google.com/citations?user=gBkYZeMAAAAJ), [Peng Zhang](https://scholar.google.com/citations?user=QTgxKmkAAAAJ&hl=zh-CN), [Xusen Sun](https://dblp.org/pid/308/0824.html), [Longhao Zhang](https://scholar.google.com/citations?user=qkJD6c0AAAAJ), [Shiqi Yang](https://www.shiqiyang.xyz/), [Bang Zhang](https://dblp.org/pid/11/4046.html), [Liefeng Bo](https://scholar.google.com/citations?user=FJwtMf0AAAAJ&hl=zh-CN), [Yaxing Wang](https://scholar.google.es/citations?user=6CsB8k0AAAAJ), [Yaxing Wang](https://scholar.google.es/citations?user=6CsB8k0AAAAJ), [Ming-Ming Cheng](https://mmcheng.net/cmm)



<div align="center">
<a href='https://montaellis.github.io/mate3d/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &ensp;
<a href='https://arxiv.org/abs/2312.06947'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> &ensp;
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/zMNYan1mIds) &ensp;
<a href='https://huggingface.co/datasets/Ellis/CatMaskHQ'><img src='https://img.shields.io/static/v1?label=Dataset&message=HuggingFace&color=yellow'></a> &ensp;
<a href='https://huggingface.co/Ellis/MaTe3D'><img src='https://img.shields.io/static/v1?label=Models&message=HuggingFace&color=yellow'></a> &ensp;
</div>

![Teaser Image](docs/teaser.png "Teaser")


## MaTe3D

3D-aware portrait editing has a wide range of applications in multiple fields. However, current approaches are limited due that they can only perform mask-guided or text-based editing. Even by fusing the two procedures into a model, the editing quality and stability cannot be ensured. To address this limitation, we propose \textbf{MaTe3D}: mask-guided text-based 3D-aware portrait editing. In this framework, first, we introduce a new SDF-based 3D generator which learns local and global representations with proposed SDF and density consistency losses. This enhances masked-based editing in local areas; second, we present a novel distillation strategy: Conditional Distillation on Geometry and Texture (CDGT). Compared to exiting distillation strategies, it mitigates visual ambiguity and avoids mismatch between texture and geometry, thereby producing stable texture and convincing geometry while editing. Additionally, we create the CatMask-HQ dataset, a large-scale high-resolution cat face annotation for exploration of model generalization and expansion. We perform expensive experiments on both the FFHQ and CatMask-HQ datasets to demonstrate the editing quality and stability of the proposed method. Our method faithfully generates a 3D-aware edited face image based on a modified mask and a text prompt.

## Preparation
### Environment
```bash
pip install torch torchvision
```

### Dataset
Download [CatMaskHQ](https://huggingface.co/datasets/Ellis/CatMaskHQ) dataset.

To expand the scope beyond human face and explore the model generalization and expansion, we design the CatMask-HQ dataset with the following representative features:

**Specialization**:  CatMask-HQ is specifically designed for cat faces, including precise annotations for six facial parts (background, skin, ears, eyes, nose, and mouth) relevant to feline features.
**High-Quality Annotations**: The dataset benefits from manual annotations by $50$ annotators and undergoes $3$ accuracy checks, ensuring high-quality labels and reducing individual differences.
**Substantial Dataset Scale**: With approximately $5,060$ high-quality real cat face images and corresponding annotations, CatMask-HQ provides ample training database for deep learning models.

![CatMaskHQ Image](docs/cat.png "CatMaskHQ")


### Model Weight
| Model | Data |  URL   |
|-------|------|--------|
| MaTe3D-Face | FFHQ-Mask | [:link:](xxx) |
| MaTe3D-CatFace | CatMaskHQ | [:link:](xx) |

## Inference
Stay tuned!



## Citation	

```
@article{zhou2023mate3d,
  title     = {MaTe3D: Mask-guided Text-based 3D-aware Portrait Editing},
  author    = {Kangneng Zhou, Daiheng Gao, Xuan Wang, Jie Zhang, Peng Zhang, Xusen Sun, Longhao Zhang, Shiqi Yang, Bang Zhang, Liefeng Bo, Yaxing Wang, Ming-Ming Cheng},
  journal   = {arXiv preprint arXiv:2312.06947},
  website   = {https://montaellis.github.io/mate-3d/},
  year      = {2023}}
```
