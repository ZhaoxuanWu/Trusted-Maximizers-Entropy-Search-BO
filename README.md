# Trusted-Maximizers Entropy Search for Efficient Bayesian Optimization

This repository is the official implementation of the following paper accepted by the Conference on Uncertainty in Artificial Intelligence (UAI) 2021:

> Quoc Phong Nguyen*, Zhaoxuan Wu*, Bryan Kian Hsiang Low, Patrick Jaillet
>
> Trusted-Maximizers Entropy Search for Efficient Bayesian Optimization

## Requirements

To install requirements:
```setup
pip install cmake
pip install -r requirements.txt
```

## Running the scripts
As an example, optimizing the synthetic Branin function, run
```bash
bash script_batch_branin.sh
```

To optimize real-world neural architecture search for CIFAR-10, run
```bash
bash script_batch_cifar.sh
```

To optimize real-world optimization problem of synthesizing faces to fool the [python face_recognition library](https://github.com/ageitgey/face_recognition), first download the multimodal discriminant analysis (MMDA) faces synthesizer model from [here](https://drive.google.com/file/d/1eOrXcWcU8YDxefTvJ2Nz2-u74IDbbS6y/view?usp=sharing), put the model under the `./face_attack/models/` directory, then run 
```bash
bash script_batch_face.sh
```

Configurations in the `.sh` files can be changed to fit different purposes. Some other pre-defined functions can be found in the [functions.py](functions.py) file.


## Remarks
In this code repository, we implement:

- TES_sp: the configurations are `criterion=='sftl'` and `mode=='sample'`;
- TES_ep: the configurations are `criterion=='ftl'` and `mode=='ep'`. However, the approximation with EP by matching the moments occasionally encounters numerical issues. Alternatively, we could resort to using samples to compute the moments: `criterion=='ftl'` and `mode=='empirical'` instead.