import os
import argparse

import wandb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download MNIST & save as a wandb artifact')
    parser.add_argument('entity', help='wandb entity (usually your username)')
    parser.add_argument('project', help='wandb project')
    args = parser.parse_args()
    
    if not os.path.exists('./tmp/MNIST'):
        os.makedirs('./temp/')
        os.system('wget www.di.ens.fr/~lelarge/MNIST.tar.gz -o ./tmp/MNIST.tar.gz')
        os.system('tar -zxvf ./tmp/MNIST.tar.gz -C ./tmp/')
    
    wandb.login()