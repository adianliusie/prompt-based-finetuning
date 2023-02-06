import random

from copy import deepcopy
from typing import List, Dict, Tuple, TypedDict
from datasets import load_dataset
from .load_classification_hf import _create_splits, _rename_keys

class TextPair(TypedDict):
    """Output example formatting (only here for documentation)"""
    text_1 : str
    text_2 : str
    label  : int 

#== main loading function ==============================================================================# 
HF_NLI_DATA = ['snli', 'mnli', 'hans', 'anli']
HF_PARA_DATA = ['qqp', 'mrpc', 'paws']

def load_hf_pair_data(data_name):
    """ loading NLI datsets available on huggingface hub """
    if   data_name == 'snli'  : train, dev, test = load_snli()
    elif data_name == 'mnli'  : train, dev, test = load_mnli()
    elif data_name == 'mnli-u': train, dev, test = load_mnli_unmatched()
    elif data_name == 'hans'  : train, dev, test = load_hans()
    elif data_name == 'anli'  : train, dev, test = load_anli(v=1)
    elif data_name == 'qqp'   : train, dev, test = load_qqp()
    elif data_name == 'mrpc'  : train, dev, test = load_mrpc()
    elif data_name == 'paws'  : train, dev, test = load_paws()
    else: raise ValueError(f"invalid text pair dataset name: {data_name}")
    return train, dev, test

#== NLI dataset loader ============================================================================#
def load_snli()->Tuple[List[TextPair], List[TextPair], List[TextPair]]:
    def _filter(data_split):
        return [i for i in data_split if i['label'] != -1]
        
    dataset = load_dataset("snli")
    train = list(dataset['train'])
    dev   = list(dataset['validation'])
    test  = list(dataset['test'])
    
    train, dev, test = [_filter(data) for data in [train, dev, test]]
    train, dev, test = _rename_keys(train, dev, test, old_key='premise',    new_key='text_1')
    train, dev, test = _rename_keys(train, dev, test, old_key='hypothesis', new_key='text_2')
    return train, dev, test

def load_mnli()->Tuple[List[TextPair], List[TextPair], List[TextPair]]:
    dataset = load_dataset('glue', 'mnli')
    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test       = list(dataset['validation_matched'])
    
    train, dev, test = _rename_keys(train, dev, test, old_key='premise',    new_key='text_1')
    train, dev, test = _rename_keys(train, dev, test, old_key='hypothesis', new_key='text_2')
    return train, dev, test

def load_mnli_unmatched()->Tuple[List[TextPair], List[TextPair], List[TextPair]]:
    dataset = load_dataset('glue', 'mnli')
    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test       = list(dataset['validation_unmatched'])
    
    train, dev, test = _rename_keys(train, dev, test, old_key='premise',    new_key='text_1')
    train, dev, test = _rename_keys(train, dev, test, old_key='hypothesis', new_key='text_2')
    return train, dev, test

def load_hans()->Tuple[List[TextPair], List[TextPair], List[TextPair]]:
    dataset = load_dataset("hans")
    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test  = list(dataset['validation'])
    
    train, dev, test = _rename_keys(train, dev, test, old_key='premise',    new_key='text_1')
    train, dev, test = _rename_keys(train, dev, test, old_key='hypothesis', new_key='text_2')
    return train, dev, test

def load_anli(v=1)->Tuple[List[TextPair], List[TextPair], List[TextPair]]:
    dataset = load_dataset("anli")
    train = list(dataset[f'train_r{v}'])
    dev   = list(dataset[f'dev_r{v}'])
    test  = list(dataset[f'test_r{v}'])

    train, dev, test = _rename_keys(train, dev, test, old_key='premise',    new_key='text_1')
    train, dev, test = _rename_keys(train, dev, test, old_key='hypothesis', new_key='text_2')
    return train, dev, test

#== paraphrasing dataset loaders ==================================================================#
def load_qqp()->Tuple[List[TextPair], List[TextPair], List[TextPair]]:
    dataset = load_dataset('glue', 'qqp')

    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.9)
    test = list(dataset['validation'])
    _, test = _create_splits(test, 0.75) # take 10_000 examples from validation as test

    train, dev, test = _rename_keys(train, dev, test, old_key='question1', new_key='text_1')
    train, dev, test = _rename_keys(train, dev, test, old_key='question2', new_key='text_2')
    return train, dev, test

def load_mrpc()->Tuple[List[TextPair], List[TextPair], List[TextPair]]:
    dataset = load_dataset('glue', 'mrpc')

    train = list(dataset[f'train'])
    dev   = list(dataset[f'validation'])
    test  = list(dataset[f'test'])

    print(len(train), len(dev), len(test))
    train, dev, test = _rename_keys(train, dev, test, old_key='sentence1', new_key='text_1')
    train, dev, test = _rename_keys(train, dev, test, old_key='sentence2', new_key='text_2')
    return train, dev, test

def load_paws()->Tuple[List[TextPair], List[TextPair], List[TextPair]]:
    dataset = load_dataset("paws", 'labeled_final')
    train = list(dataset['train'])
    dev   = list(dataset['validation'])
    test  = list(dataset['test'])
    train, dev, test = _rename_keys(train, dev, test, old_key='sentence1', new_key='text_1')
    train, dev, test = _rename_keys(train, dev, test, old_key='sentence2', new_key='text_2')
    return train, dev, test

