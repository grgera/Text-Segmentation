import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import f1_score, accuracy_score

class BERT_Model(pl.LightningModule):
    def __init__(self, model, emb_type, freeze_bert, lr, class_w, num_classes):
        super().__init__()

        self.emb_type = emb_type
        self.num_classes = num_classes
        D_in, H1, H2 = 768, 512, 256

        self.bert = model.bert.encoder
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H1),
            nn.GELU(),
            nn.Linear(H1, H2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(H2, num_classes)
        )

        self.freeze_params(freeze_bert)

        if emb_type == 'SBERT':
            input_size, hid_dim = 384, 768
            self.fc = nn.Linear(input_size, hid_dim)

        if class_w is not None:
            self.loss_func = nn.CrossEntropyLoss(weight=class_w,reduction='mean')
        else:
            self.loss_func = nn.CrossEntropyLoss()

        self.learning_rate = lr
        self.save_hyperparameters()
    
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        """
        if self.emb_type == 'SBERT':
            input_ids = self.fc(input_ids)

        input = torch.unsqueeze(input_ids, dim=0)
        attention = torch.unsqueeze(attention_mask, dim=0)
        outputs = self.bert(input, attention)
        logits = self.classifier(torch.squeeze(outputs[0]))

        return logits

    def freeze_params(self, freeze):
        """
        Function for the possibility of separate training of the "head" and "body" of the BERT-like model.
        """
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        if not freeze:
            for param in self.bert.parameters():
                param.requires_grad = True

    def training_step(self, batch, batch_idx):
        input, attn, label = batch
        logits = self(input, attn)
        loss = self.loss_func(logits, label)
        self.log('train_loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        input, attn, label = batch
        logits = self(input, attn)

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
        input, attn, label = batch
        logits = self(input, attn)

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