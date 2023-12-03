# Unstructured_Relu_Pruning_Reduce_Depth

## Notice

If you want to use our code to launch experiments with CIFAR-10 dataset, please make sure you have downloaded the 'cifar10_models' folder in 'src'.
To launch experiments with Tiny-Imagenet dataset, make sure you already have Tiny-Imagenet dataset on your device. 


## Pruning
```bash
python3 src/main.py
```



## Reinilize and finetune
```bash
python3 src/reinitialize_finetune.py
```

reinitialize the EGP model and finetune it.


## Calculate Entropy and generate histogram

```bash
python3 src/generate_histogram.py
```
