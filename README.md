# Pytorch Baseline Code with horovod
Pytorch ImageNet training baseline code with horovod for DDP (Distributed Data Parallel) and Wandb.

## Requirements
- Install PyTorch ([pytorch.org](http://pytorch.org))  
- Install Horovod ([Horovod](https://github.com/horovod/horovod#install))
- `pip install -r requirements.txt`

## Training
```bash
horovodrun -np 8 python3 main.py -a resnet50 --distributed /dataset/ImageNet
```
