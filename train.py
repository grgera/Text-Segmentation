import re
import yaml
import torch
import warnings
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
import pytorch_lightning as pl
from transformers import logging
import torch.nn.functional as F
from pytorch_lightning import Trainer
import os, argparse, glob, itertools, random
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sentence_transformers import SentenceTransformer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel

warnings.filterwarnings('ignore')
logging.set_verbosity_error()

from labse import LaBSE_Embedding
from sbert import SBERT_Embedding
from data_module import DataModule
from bert import BERT_Model
from bilstm import BILSTM_Model
from utils import *

def _train_stage(full_model, data, epochs, log, callb):
    """
    Defining and launching training using Trainer API.
    """
    trainer = Trainer(accelerator="gpu", devices=1, max_epochs=epochs, logger=log, callbacks=callb)
    trainer.fit(full_model, data)

def _sbert_preprocess(args, cnf, num_classes, dataframe):
    """
    Compute sentence embeddings and make dataset with them.
    """
    class_weight = cnf["class_weight"]
    sbert_model = SentenceTransformer(cnf['st_checkpoint'])
    sbert_emb = SBERT_Embedding(dataframe, sbert_model, batch_size=cnf['batch_size'], binary_mode=args.binary_mode)

    if num_classes == 2:
        class_weight = sbert_emb.get_class_weight()

    train_dataset, val_dataset, test_dataset = sbert_emb.get_dataset('train'), sbert_emb.get_dataset('eval'), sbert_emb.get_dataset('test')
    datasets = [train_dataset, val_dataset, test_dataset]
    return datasets, class_weight

def _labse_preprocess(args, cnf, num_classes, dataframe):
    """
    Compute sentence embeddings and make dataset with them.
    """
    class_weight = cnf["class_weight"]
    labse_tokenizer = AutoTokenizer.from_pretrained(cnf['lb_checkpoint'])
    labse_model = AutoModel.from_pretrained(cnf['lb_checkpoint'])

    labse_emb = LaBSE_Embedding(dataframe, labse_model, labse_tokenizer, batch_size=cnf['batch_size'], binary_mode=args.binary_mode)

    if num_classes == 2:
        class_weight = labse_emb.get_class_weight()
    
    datasets = labse_emb.get_datasets()

    return datasets, class_weight

def train(args, config):
    dataframe = get_data()
    wdb_config = config['wandb']

    wandb_logger = WandbLogger(
        save_dir=wdb_config["save_dir"],
        project=wdb_config["project"],
        log_model=True,
        offline=wdb_config["offline"],
        config=config
    )

    callbacks = ModelCheckpoint(dirpath=wdb_config['checkpoint_directory'],
                                filename='{}-{}-{}'.format(args.emb_type, args.sec_model, args.binary_mode),
                                monitor="val_f1",
                                save_top_k=1, mode='max')

    progress_bar = TQDMProgressBar(refresh_rate=wdb_config["progress_bar_refresh_rate"])

    num_classes = 3 if args.binary_mode == 'None' else 2

    if args.emb_type == 'SBERT':
        cnf = config['sbert']
        datasets, class_w = _sbert_preprocess(args, cnf, num_classes, dataframe)

    if args.emb_type == 'LABSE':
        cnf = config['labse']
        datasets, class_w = _labse_preprocess(args, cnf, num_classes, dataframe) 

    if args.sec_model == 'BERT':
        bert_base = AutoModelForTokenClassification.from_pretrained(cnf['bert_checkpoint'])
        data = DataModule(datasets, cnf['batch_size'])
        full_model = BERT_Model(bert_base, cnf['emb_type'], cnf['freeze_bert'], cnf['learning_rate'], class_w, num_classes)

        print('Training process starts!\n')
        _train_stage(full_model, data, epochs=7, log=wandb_logger, callb=[callbacks, progress_bar])
        full_model.freeze_params(freeze=False)
        _train_stage(full_model, data, epochs=12, log=wandb_logger, callb=[callbacks, progress_bar])
    
    if args.sec_model == 'LSTM':
        cnf = config["bilstm"]
        data = DataModule(datasets, config["labse"]["batch_size"])
        full_model = BILSTM_Model(**cnf, num_classes=num_classes)

        print('Training process starts!\n')
        _train_stage(full_model, data, epochs=5, log=wandb_logger, callb=[callbacks, progress_bar])

def main():
    print('Training preprocessing...\n')

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='')
    parser.add_argument('--emb_type', default='')
    parser.add_argument('--binary_mode', default='None')
    parser.add_argument('--sec_model', default='BERT')

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    torch.manual_seed(config['seed'])
    train(args, config)

if __name__ == '__main__':
    main() 