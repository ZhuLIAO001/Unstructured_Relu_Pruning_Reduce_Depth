# Unstructured_Relu_Pruning_Reduce_Depth

## Notice

If you want to use our code to launch experiments with CIFAR-10 dataset, please make sure you have downloaded the 'cifar10_models' folder in 'src'.
To launch experiments with Tiny-Imagenet dataset, make sure you already have Tiny-Imagenet dataset on your device. 


## Pruning

- `Prun_Dataset_Model.py` corresponding to different Pruning composition.

-baseline means the code implement traditional unstuctured iterative pruning for the whole model. -EGP means the code implement EGP with ReLU/GeLU activated layers. 
For Swin-T related codes, -GeluActi_layer means the code implement traditional unstuctured iterative pruning for only the GeLU activated layers.

For example:

```bash
python Prun_cifar10_ResNet18_EGP.py
```

launch the EGP with Resnet-18 model on Cifar10 dataset.


## Reinilize and finetune

- 'Dataset_Model_REinial_finetune.py' is used to reinitialize corresponding EGP model and finetune it.

For example:

```bash
python CIFAR10_SwinT_REinial_finetune.py
```

reinitialize the EGP model of Swin-T trained on Cifar10 and finetune it.


## Calculate Entropy and generate histogram

- 'Entropy_Histogram_Model_Dataset.py' is used to calculate the Entropy of ReLU/GeLU layers' input and generate the histogram.

For example:

```bash
python Entropy_Histogram_Resnet18_cifar10.py
```

calculate the entropy of ReLU layers' input in Resnet18 model trained on Cifar10 dataset and generate the histogram.
