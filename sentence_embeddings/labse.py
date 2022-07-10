
import re
import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class LaBSE_Embedding(nn.Module):
    def __init__(self, data, model, tokenizer, batch_size, binary_mode):
        super().__init__()
        self.device = 'cuda'
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.data = data
        self.batch_size = batch_size
        self.binary = binary_mode

        self.train_split()

    def text_preprocessing(self, s):
        """
        Lowercase the sentence, remove "@name", isolate and remove punctuations except "?"
        Remove other special characters, remove trailing whitespace
        """
        s = s.lower()
        s = re.sub(r'(@.*?)[\s]', ' ', s)
        s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
        s = re.sub(r'[^\w\s\?]', ' ', s)
        s = re.sub(r'([\;\:\|•«\n])', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()

        return s
    
    def get_class_weight(self):
        """
        Compute class weights among labels for imbalanced data.
        """
        y = torch.cat((self.y_train, self.y_val, self.y_test), dim=0)
        class_weights=compute_class_weight('balanced', classes=np.unique(y), y=np.array(y))

        return torch.tensor(class_weights, dtype=torch.float)

    def binary_mode(self, df_train, df_val, df_test):
        """
        Compute labels corresponding to the selected classification method.
        """
        if self.binary == 'None':
            return torch.tensor(df_train[1].values.tolist()), \
                   torch.tensor(df_val[1].values.tolist()), \
                   torch.tensor(df_test[1].values.tolist())

        if self.binary == 'paragraph':
            self.num = 2
            y_train = np.where(df_train[1].values == self.num, 0, df_train[1].values.tolist())
            y_val = np.where(df_val[1].values == self.num, 0, df_val[1].values.tolist())
            y_test = np.where(df_test[1].values == self.num, 0, df_test[1].values.tolist())

            return torch.tensor(y_train), torch.tensor(y_val), torch.tensor(y_test)

        if self.binary == 'section':
            self.num = 1
            y_train = np.where(df_train[1].values == self.num, 0, df_train[1].values.tolist())
            y_val = np.where(df_val[1].values == self.num, 0, df_val[1].values.tolist())
            y_test = np.where(df_test[1].values == self.num, 0, df_test[1].values.tolist())

            return torch.tensor(np.where(y_train == 2, 1, y_train)), \
                   torch.tensor(np.where(y_val == 2, 1, y_val)), \
                   torch.tensor(np.where(y_test == 2, 1, y_test))

    def train_split(self):
        """
        Splitting data into three usual parts.
        """
        df_train, df_test = train_test_split(self.data, random_state=42, test_size=0.15, shuffle=False)
        df_train, df_val = train_test_split(df_train, random_state=42, test_size=0.1, shuffle=False)

        self.y_train, self.y_val, self.y_test = self.binary_mode(df_train, df_val, df_test)

        self.X_train = list(map(self.text_preprocessing, df_train[0].values))
        self.X_val = list(map(self.text_preprocessing, df_val[0].values))
        self.X_test = list(map(self.text_preprocessing, df_test[0].values))

    def get_loader(self, text, labels, max_len):
        """
        Get DataLoader for compute sentence embeddings.
        """
        input_ids = []
        attention_masks = []

        for sent in text:
            encoded = self.tokenizer.encode_plus(sent, add_special_tokens=True, max_length=max_len, pad_to_max_length=True, return_tensors='pt', return_attention_mask=True)
            input_ids.append(encoded.get('input_ids'))
            attention_masks.append(encoded.get('attention_mask'))

        dataset = TensorDataset(torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0), labels)
        dataloader = DataLoader(dataset, shuffle=False, batch_size=self.batch_size)

        return dataloader

    def get_embedings(self, data, max_length):
        """
        Compute sentence embeddings from input data.
        """
        if data == 'train':
            dataloader = self.get_loader(self.X_train, self.y_train, max_length)
        if data == 'eval':
            dataloader = self.get_loader(self.X_val, self.y_val, max_length)
        if data == 'test':
            dataloader = self.get_loader(self.X_test, self.y_test, max_length)

        labels = []
        features = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = [r.to(self.device) for r in batch]

                model_output = self.model(batch[0], batch[1])
                embeddings = model_output.pooler_output
                embeddings = F.normalize(embeddings)

                labels.append(batch[2])
                features.append(embeddings)
        
        return torch.cat(features, dim=0), torch.cat(labels, dim=0)

    def get_datasets(self):
        """
        Packing data into TensorDataset type.
        """
        train_features, train_labels = self.get_embedings('train', 128)
        val_features, val_labels = self.get_embedings('eval', 128)
        test_features, test_labels = self.get_embedings('test', 128)

        train_mask = torch.FloatTensor(np.ones(train_features.shape[0]))
        val_mask = torch.FloatTensor(np.ones(val_features.shape[0]))
        test_mask = torch.FloatTensor(np.ones(test_features.shape[0]))

        train_dataset = TensorDataset(train_features, train_mask, train_labels)
        val_dataset = TensorDataset(val_features, val_mask, val_labels)
        test_dataset = TensorDataset(test_features, test_mask, test_labels)

        return [train_dataset, val_dataset, test_dataset]