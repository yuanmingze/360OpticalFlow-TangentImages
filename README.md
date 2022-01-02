 **360° Optical Flow using Tangent Images**

[Video](https://youtu.be/RczNEQefOTo) | [Paper](https://arxiv.org/pdf/2112.14331.pdf)

# 1. Introduction

This repository contains the source code for BMVC 2021 paper "360° Optical Flow using Tangent Images".

# 2. Running code

In this project root folder run `pip install -r requirements.txt` to install the dependency.

Than run `python main.py` to estimate the 360° optical flow with demo data.

# 3. Datasets

- Replica 360°

Please download the Replica 360° optical flow data from [link](https://drive.google.com/file/d/14O9jP5dknguXMhAsSD5yq1_4tv3pJvm5/view?usp=sharing).

The mask file `xxxx_mask_pano.png` indicate the unavailable pixels.
The value 0 pixel's RGB, depth and optical flow are unavailable, the value 1 pixels are available.

- OmniPhotos

Please download the 360° OmniPhotos dataset from [OmniPhotos webpage](https://richardt.name/publications/omniphotos/).

# 4. Citation

If you think our code useful, please consider citing our paper:

```bash
@InProceedings{yuan2021panoof,
  title={360° Optical Flow using Tangent Images},
  author={Yuan, Mingze and Christian, Richardt},
  booktitle = {Proceedings of the 32nd British Machine Vision Conference (BMVC)},
  year={2021}
}
```

# 5. Acknowledgements

We thank the reviewers for their valuable feedback that has helped us improve our paper.

This work was supported by an EPSRC-UKRI Innovation Fellowship (EP/S001050/1) and RCUK grant CAMERA (EP/M023281/1, EP/T022523/1).

# License
This repository is released under the MIT license.
