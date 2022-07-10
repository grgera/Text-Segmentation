import json
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def read_json(fname):
    """
    Read and write json type file to list.
    """
    data = [json.loads(line) for line in open(fname, 'r')]
    return data

def dt_to_pd(dataset):
    """
    Сonvert dataset to DataFrame.
    """
    news_list = [pd.DataFrame(item) for item in dataset]
    df = pd.concat(news_list, ignore_index=True)

    return df

def get_data():
    """
    Read json file and return DataFrame
    """
    data_list = read_json('data.jsonl')
    dataframe = dt_to_pd(data_list)
    return dataframe

def words_in_sentences(datasets):
    """
    A function for counting word counts in sentences.
    """
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(16, 6), dpi=100)
     
    sns.histplot([len(sent.split()) for sent in datasets[0].values.tolist()], 
                     ax=axes, 
                     multiple="stack", 
                     palette="Blues_r", 
                     edgecolor=".3", 
                     linewidth=.5,)
    axes.set_xlabel("Number of words in sentence")
    axes.set_ylabel("Number of samples")

    fig.suptitle("Sentence Length Distribution", va='baseline', fontsize=15)
    axes.set_title("Data")

    plt.show();