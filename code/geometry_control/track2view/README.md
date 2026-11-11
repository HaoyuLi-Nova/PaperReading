# Track2View: 4D-Consistent Camera-Controlled Video Generation via Paired 3D Point Tracks

<div align="center">

<a href="https://qjizhi.github.io/track2view/"><img src="https://img.shields.io/badge/Project_Page-Track2View-1a56db?style=flat-square" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2606.15534"><img src="https://img.shields.io/badge/arXiv-2606.15534-b31b1b?style=flat-square" alt="arXiv"></a>

<img src="assets/teaser.gif" width="85%"/>

</div>

> **Re-render any video under arbitrary target camera trajectories — with explicit, temporally continuous source-to-target correspondence.**

**[Feng Qiao](https://qjizhi.github.io/)<sup>1</sup>, [Zhaochong An](https://zhaochongan.github.io/)<sup>2</sup>, [Zhexiao Xiong](https://steven-xiong.github.io/)<sup>1</sup>, [Serge Belongie](https://belongielab.org/)<sup>2</sup>, [Nathan Jacobs](https://jacobsn.github.io/)<sup>1</sup>**

<sup>1</sup>Washington University in St. Louis &nbsp;&nbsp; <sup>2</sup>University of Copenhagen

---

## 📢 News

- **[2026.06]** Project page and paper are now available. **Code and data annotations are coming soon.**

## 📖 Abstract

Re-rendering an existing video from a novel camera viewpoint requires the output to follow the
prescribed camera trajectory while preserving the appearance and dynamics of the original scene across
every frame. Existing methods rely on per-frame pose embeddings, noisy point-cloud renderings, or
implicit learned correspondences, none of which provides an explicit, temporally continuous link between
source and target pixels. We propose **Track2View**, which conditions a video diffusion transformer on
**paired 3D point tracks**: sparse trajectories of scene points projected into both the source and
target camera views. These tracks provide explicit spatiotemporal correspondences that are temporally
continuous by construction, encoding what content should appear where and when. At the core of Track2View
is a *dual-view track conditioner* that transfers visual context from source to target view through
parameter-free geometric operations and learned temporal aggregation, ensuring generalization to
arbitrary camera trajectories without memorizing specific motions. We further introduce a data curation
pipeline that extracts one-to-one track correspondences by running a 3D point tracker on temporally
concatenated multi-camera view pairs. On a 400-video benchmark spanning static and dynamic scenes,
Track2View achieves state-of-the-art results across visual quality, view synchronization, and camera
accuracy, reducing rotation error by **30–65%** and translation error by **61–72%** relative to leading
baselines.

## 🚧 TODO

- [ ] Release inference code and pretrained checkpoints
- [ ] Release training code
- [ ] Release data annotations (paired 3D point tracks)

## 🌟 Citation

If you find our work helpful, please consider leaving a star and citing our paper:

```bibtex
@article{track2view2026,
  title   = {Track2View: 4D-Consistent Camera-Controlled Video Generation
             via Paired 3D Point Tracks},
  author  = {Feng Qiao and Zhaochong An and Zhexiao Xiong and Serge Belongie and Nathan Jacobs},
  journal = {arXiv preprint arXiv:2606.15534},
  year    = {2026}
}
```
