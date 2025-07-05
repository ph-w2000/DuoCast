## Code

### arXiv Link

Please visit [arXiv](https://arxiv.org/pdf/2412.01091) for more information.

### Environment

```shell
conda env create -f env.yaml
conda activate DuoCast
```
### Resource
Pretrained DuoCast_SimVP: [Google Drive](https://drive.google.com/file/d/1gm1gHCSC0qgH9oqKcF-W4M3YZ13fJRJt/view?usp=share_link)
Pretrained DuoCast_PhDNet: [Google Drive](https://drive.google.com/file/d/1WS9pu6Ssde1hNPC1wIEO8Qyhktu7hFem/view?usp=share_link)
Pretrained Autoencoder: [Google Drive](https://drive.google.com/file/d/1bA63-3UV-uVnVDXdpEBzsV9ekz_8ZpKm/view?usp=share_link)

### Evaluation
```shell
# Note: Config the dataset path in `dataset/get_dataset.py` before running.
python run.py --eval --ckpt_milestone Exps/basic_exps/DuoCast_sevir_None/checkpoints/ckpt-170080.pt
```
### Backbone Training
```shell
# Note: Need a checkpoint from stage1 PrecipFlow model.
python run.py --ckpt_milestone Exps/basic_exps/DuoCast_sevir_None/checkpoints/ckpt-170080.pt
```

### Continue Training
```shell
python run.py --continue_train --ckpt_milestone Exps/basic_exps/DuoCast_sevir_None/checkpoints/ckpt-170080.pt
```

### Display Video

You can view the video by downloading it [here](resources/display_video.mp4).
