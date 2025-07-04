#!/usr/bin/python3
import argparse
import os
from trainer import Eva_Trainer, Reg_Trainer
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def main():

    # load config 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/data/zcl3/CVPR2025/Major_revision/code/Our/Our_os/Yaml/Registartion.yaml', help='Path to the config file.')
    parser.add_argument('--train', action='store_true',default=False, help='Train mode')
    opts = parser.parse_args()
    config = get_config(opts.config)
    print(config)
 

    # Tranin Evaluator  
    #  
    if 'Evaluator' in config['name']:
        trainer = Eva_Trainer(config)
        if opts.train:
            trainer.train()
        else:
            trainer.test()

    # Tranin Registration Network    
    #    
    elif 'Registration' in config['name']:
        trainer = Reg_Trainer(config)
        if opts.train:
            trainer.train()
        else:
            trainer.test()
    else:
        raise Exception('Unknown config file.')


if __name__ == '__main__':
    main()
