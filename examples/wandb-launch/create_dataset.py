import os
import argparse

import wandb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download MNIST & save as a wandb artifact')
    parser.add_argument('entity', help='wandb entity (usually your username)')
    parser.add_argument('project', help='wandb project')
    args = parser.parse_args()
    
    if not os.path.exists('./temp/MNIST'):
        if not os.path.exists('./temp'):
            os.makedirs('./temp') 
        os.system('wget www.di.ens.fr/~lelarge/MNIST.tar.gz --directory-prefix ./temp')
        os.system('tar -zxvf ./temp/MNIST.tar.gz -C ./temp/')
        os.system('rm ./temp/MNIST.tar.gz')

    wandb.login()
    
    config = {'dataset':'MNIST'}
    with wandb.init(entity=args.entity, project=args.project, config=config) as run:
        data_artifact = wandb.Artifact('MNIST', 'dataset')
        data_artifact.add_dir('./temp')
        run.log_artifact(data_artifact)