# On the Diversity of Adversarial Ensemble Learning

The implementation of AdvOAP in paper "On the Diversity of Adversarial Ensemble Learning".

## Showing the Diversity
```
python main.py --mode show_diversity
```

## Training
### Train model for MNIST
```
python main.py --num_epochs 60 --num_models 3 --batch_size 512 --learning_rate 0.01 --dataset MNIST --perturbation_size 0.2 --perturb_steps_train 10 --perturb_steps_test 20 --step_size_train 0.04 --step_size_test 0.02
```

### Train model for FMNIST
```
python main.py --num_epochs 60 --num_models 3 --batch_size 512 --learning_rate 0.01 --dataset Fashion-MNIST --perturbation_size 0.05 --perturb_steps_train 10 --perturb_steps_test 20 --step_size_train 0.01 --step_size_test 0.005
```

### Train model for CIFAR10
```
python main.py --num_epochs 250 --num_models 3 --batch_size 256 --learning_rate 0.1 --dataset CIFAR10 --perturbation_size 0.031 --perturb_steps_train 10 --perturb_steps_test 20 --step_size_train 0.0078 --step_size_test 0.0039
```

## Testing
We provide the trained model in checkpoint folder for CIFAR10, run the following command for testing
```
python main.py --mode test
```
