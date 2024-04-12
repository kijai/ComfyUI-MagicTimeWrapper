# ComfyUI wrapper node for testing MagicTime

# UPDATE

While making this I figured out that I could just extract the lora and apply it to the v3 motion model to use it as it is with any Animatediff-Evolved workflow, the merged v3 checkpoint along with the spatial lora converted to .safetensors, are available here:

https://huggingface.co/Kijai/MagicTime-merged-fp16

**This does NOT need this repo, I will not be updating this further.**

___

## Only use this repo and the following instructions for legacy/testing purposes:

https://github.com/kijai/ComfyUI-MagicTimeWrapper/assets/40791699/c71d271d-8219-456c-891d-da9bdbd44d54

# Installing
Either use the Manager and it's install from git -feature, or clone this repo to custom_nodes and run:

`pip install -r requirements.txt`

or if you use portable (run this in ComfyUI_windows_portable -folder):

`python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-MagicTimeWrapper\requirements.txt`

You can use any 1.5 model, and the v3 AnimateDiff motion model 
placed in `ComfyUI/models/animatediff_models`: 

https://huggingface.co/guoyww/animatediff/blob/main/v3_sd15_mm.ckpt

rest (131.0 MB) is **auto downloaded**, from https://huggingface.co/BestWishYsh/MagicTime/tree/main/Magic_Weights
to `ComfyUI/modes/magictime` 
___
# Original repo:
https://github.com/PKU-YuanGroup/MagicTime


## 🐳 ChronoMagic Dataset
ChronoMagic with 2265 metamorphic time-lapse videos, each accompanied by a detailed caption. We released the subset of ChronoMagic used to train MagicTime. The dataset can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1WsomdkmSp3ql3ImcNsmzFuSQ9Qukuyr8?usp=sharing). Some samples can be found on our Project Page.


## 👍 Acknowledgement
* [Animatediff](https://github.com/guoyww/AnimateDiff/tree/main) The codebase we built upon and it is a strong U-Net-based text-to-video generation model.

* [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan) The codebase we built upon and it is a simple and scalable DiT-based text-to-video generation repo, to reproduce [Sora](https://openai.com/sora).

## 🔒 License
* The majority of this project is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/PKU-YuanGroup/MagicTime/blob/main/LICENSE) file.
* The service is a research preview intended for non-commercial use only. Please contact us if you find any potential violations.



## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```BibTeX
@misc{yuan2024magictime,
      title={MagicTime: Time-lapse Video Generation Models as Metamorphic Simulators}, 
      author={Shenghai Yuan and Jinfa Huang and Yujun Shi and Yongqi Xu and Ruijie Zhu and Bin Lin and Xinhua Cheng and Li Yuan and Jiebo Luo},
      year={2024},
      eprint={2404.05014},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 🤝 Contributors

<a href="https://github.com/PKU-YuanGroup/MagicTime/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PKU-YuanGroup/MagicTime" />
</a>
