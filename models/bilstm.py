
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import get_scheduler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score, accuracy_score

class BILSTM_Model(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim1, hidden_dim2, n_layers, bidirectional, lr, class_weight, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim1,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_dim1 * 2, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        if class_weight is not None:
            self.loss_func = nn.CrossEntropyLoss(weight=class_weight,reduction='mean')
        else:
            self.loss_func = nn.CrossEntropyLoss()

        self.learning_rate = lr
        self.save_hyperparameters()
    
    def forward(self, input_ids):
        """
        Feed input to BERT and the classifier to compute logits.
        """
        input = self.embedding(input_ids)
        packed_output, (hidden, cell) = self.lstm(input)
        cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        rel = self.relu(cat)
        dense1 = self.fc1(rel)
        drop = self.dropout(dense1)
        preds = self.fc2(drop)

        return preds

    def training_step(self, batch, batch_idx):
        input, _, label = batch
        logits = self(input.long())
        loss = self.loss_func(logits, label)
        self.log('train_loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        input, _, label = batch
        logits = self(input.long())

        loss = self.loss_func(logits, label)
        preds = torch.argmax(logits, dim=1).flatten()

        if self.num_classes > 2:
            f1 = f1_score(label.cpu().numpy(), preds.cpu().numpy(), average='micro') * 100
        else:
            f1 = f1_score(label.cpu().numpy(), preds.cpu().numpy()) * 100

        accuracy = accuracy_score(label.cpu().numpy(), preds.cpu().numpy()) * 100

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer) 
        
        return [optimizer], [scheduler]

    def test_step(self, batch, batch_idx):
        input, _, label = batch
        logits = self(input.long())

        loss = self.loss_func(logits, label)
        preds = torch.argmax(logits, dim=1).flatten()

        if self.num_classes > 2:
            f1 = f1_score(label.cpu().numpy(), preds.cpu().numpy(), average='micro') * 100
        else:
            f1 = f1_score(label.cpu().numpy(), preds.cpu().numpy()) * 100

        accuracy = accuracy_score(label.cpu().numpy(), preds.cpu().numpy()) * 100

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_f1', torch.tensor(f1), prog_bar=True)
        self.log("test_acc", accuracy, prog_bar=True)

        output = dict({
            'test_loss': loss,
            'test_f1': torch.tensor(f1),
            'test_acc': torch.tensor(accuracy)
        })
        
        return output