
import re
import torch
import numpy as np
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class SBERT_Embedding(nn.Module):
    def __init__(self, data, model, batch_size, binary_mode):
        super().__init__()

        self.model = model
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
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=np.array(y))

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

    def get_dataset(self, data_type):
        """
        Packing data into TensorDataset type.
        """
        if data_type == 'train':
            features = self.model.encode(self.X_train, batch_size=self.batch_size, show_progress_bar=True, convert_to_tensor=True)
            labels = self.y_train
        if data_type == 'eval':
            features = self.model.encode(self.X_val, batch_size=self.batch_size, show_progress_bar=True, convert_to_tensor=True)
            labels = self.y_val
        if data_type == 'test':
            features = self.model.encode(self.X_test, batch_size=self.batch_size, show_progress_bar=True, convert_to_tensor=True)
            labels = self.y_test

        mask = torch.FloatTensor(np.ones(features.shape[0]))
        dataset = TensorDataset(features, mask, labels)
        
        return dataset