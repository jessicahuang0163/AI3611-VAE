# Variational Autoencoder

This example trains a Variational Autoencoder (VAE) on MNIST. 

## Files or Folders
### exp_specs: 
It manages the configs used in the experiment. Note that the configs here are the best parameters I've found under every circumstances.

mlp1.yaml is used to train the best model of VAE when latent_dim=1; mlp2.yaml is used to train the best model of VAE when latent_dim=2

predict1.yaml and predict2.yaml are used in the visulization jupyter notebook.

### image:
The output image are stored here.
### visualize_encoder_output.ipynb:
You can get a visulization of the encoder output by using this notebook (given the trained model).
### other python files:
Code for training, testing, and generating text.

## How to run the script
```bash
python main.py           
# Reproduce the best model I've found (latent_dim=2).
```

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -e, --experiment          experiment specification file
  -g, --gpu                 gpu id
```

With these arguments, a variety of models can be tested.
As an example, you can use these commands:

```bash
python main.py -e exp_specs/mlp1.yaml -g 2
```
Note that in the visulization notebook, you should specify the model you want to test (ckpt path) and keep the parameters same in the corresponding predict yaml.

## Parameter Tuning (WandB)
Please visit https://wandb.ai/jessica-huang/VAE?workspace=user-jessica-huang for more details.