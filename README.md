## Code

### Environment

```shell
conda env create -f env.ymal
conda activate DuoCast
```

### Evaluation
```shell
# Note: Config the dataset path in `dataset/get_dataset.py` before running.
python run.py --eval --ckpt_milestone Exps/basic_exps/Diffphydnet_sevir_None/checkpoints/ckpt-170080.pt
```
### Backbone Training
```shell
python run.py 
```