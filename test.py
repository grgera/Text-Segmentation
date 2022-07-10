import yaml
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
import os, argparse, glob, itertools, random
from pytorch_lightning.loggers import WandbLogger
from sklearn.utils.class_weight import compute_class_weight

from transformers import logging
logging.set_verbosity_error()

from train import _sbert_preprocess
from train import _labse_preprocess
from bert_data import BERT_DataModule
from bert import BERT_Model
from bilstm import BILSTM_Model
from utils import *

def _test_stage(full_model, data, log):
    trainer = Trainer(accelerator="gpu", devices=1, logger=log)
    trainer.test(full_model, data)

def test(args, config):
    dataframe = get_data()
    wdb_config = config['wandb']

    wandb_logger = WandbLogger(
        save_dir=wdb_config["save_dir"],
        project=wdb_config["project"],
        log_model=True, 
        offline=wdb_config["offline"],
        config=config
    )

    num_classes = 3 if args.binary_mode == 'None' else 2

    if args.emb_type == 'SBERT':
        cnf = config['sbert']
        datasets = _sbert_preprocess(args, cnf, num_classes, dataframe)

    if args.emb_type == 'LABSE':
        cnf = config['labse']
        datasets = _labse_preprocess(args, cnf, num_classes, dataframe) 

    if args.sec_model == 'BERT':
        bert_base = AutoModelForTokenClassification.from_pretrained(cnf['bert_checkpoint'])
        data = DataModule(datasets, cnf['batch_size'])

        print('Testing process start!\n')
        model = BERT_Model.load_from_checkpoint(checkpoint_path=wdb_config["checkpoint_directory"] + '/{}-{}-{}.ckpt'.format(args.emb_type, args.sec_model, args.binary_mode))

    if args.sec_model == 'LSTM':
        data = DataModule(datasets, cnf['batch_size'])

        print('Testing process start!\n')
        model = BILSTM_Model.load_from_checkpoint(checkpoint_path=wdb_config["checkpoint_directory"] + '/{}-{}-{}.ckpt'.format(args.emb_type, args.sec_model, args.binary_mode))

    _test_stage(model, data, wandb_logger)

def main():
    print('Test preprocessing...\n')

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='')
    parser.add_argument('--emb_type', default='')
    parser.add_argument('--binary_mode', default='None')
    parser.add_argument('--sec_model', default='BERT')

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    torch.manual_seed(config['seed'])
    test(args, config)

if __name__ == '__main__':
    main() 