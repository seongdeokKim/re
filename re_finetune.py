import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch_optimizer as custom_optim

from utils.models.bert_bilstm import BERT_BiLSTM
from utils.trainer import Trainer
from utils.data_loader import BertDataset, TokenizerWrapper


def define_argparser():

    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    
    p.add_argument('--train_fn', required=True)
    p.add_argument('--test_fn', required=True)

    p.add_argument('--pretrained_model_name', type=str, default='bert-base-cased')

    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=10)

    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--warmup_ratio', type=float, default=.1)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    p.add_argument('--use_radam', action='store_true')

    p.add_argument('--max_length', type=int, default=100)
    p.add_argument('--entity_max_length', type=int, default=20)

    config = p.parse_args()

    return config


def read_file(file):

    with open(file, 'r', encoding='utf-8') as f:
        samples = []
        for line in f:
            field_line = line.strip().split("\t")

            if len(field_line) > 7:
                relation = field_line[0].strip()
                sentence = field_line[7].strip()

                left_entity = field_line[1].strip()
                left_entity_tag = "[" + field_line[2].strip().upper() + "]"
                left_entity_span = field_line[3].strip()
                right_entity = field_line[4].strip()
                right_entity_tag = "[" + field_line[5].strip().upper() + "]"
                right_entity_span = field_line[6].strip()
                d = {
                    "left_entity": left_entity,
                    "left_entity_tag": left_entity_tag,
                    "left_entity_span": left_entity_span,
                    "right_entity": right_entity,
                    "right_entity_tag": right_entity_tag,
                    "right_entity_span": right_entity_span
                }

                samples.append(
                    (relation, sentence, d)
                )

    return samples


def get_loaders(config, tokenizer):

    samples = read_file(config.train_fn)

    labels = [s[0] for s in samples]
    texts = [s[1] for s in samples]
    entity_info = [s[2] for s in samples]

    print(f'avg length of abstracts: {sum([len(text.split()) for text in texts])/len(texts)}')

    # Generate label to index map.
    unique_labels = list(set(labels))
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(unique_labels):
        label_to_index[label] = i
        index_to_label[i] = label

    # Convert label text to integer value.
    labels = list(map(label_to_index.get, labels))

    # Shuffle before split into train and validation set.
    shuffled = list(zip(texts, entity_info, labels))
    random.shuffle(shuffled)

    texts = [e[0] for e in shuffled]
    entity_info = [e[1] for e in shuffled]
    labels = [e[2] for e in shuffled]

    valid_ratio = 0.175
    idx = int(len(texts) * (1-valid_ratio))

    # Get dataloaders using given tokenizer as collate_fn.
    train_loader = DataLoader(
        BertDataset(texts[:idx], entity_info[:idx], labels[:idx]),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TokenizerWrapper(tokenizer,
                                    config.max_length,
                                    config.entity_max_length).generate_for_train,
    )
    valid_loader = DataLoader(
        BertDataset(texts[idx:], entity_info[idx:], labels[idx:]),
        batch_size=config.batch_size,
        collate_fn=TokenizerWrapper(tokenizer,
                                    config.max_length,
                                    config.entity_max_length).generate_for_train,
    )

    samples_for_test = read_file(config.test_fn)
    labels_for_test = [s[0] for s in samples_for_test]
    texts_for_test = [s[1] for s in samples_for_test]
    entity_info_for_test = [s[2] for s in samples_for_test]

    # Convert label text to integer value.
    labels_for_test = list(map(label_to_index.get, labels_for_test))
    test_loader = DataLoader(
        BertDataset(texts_for_test, entity_info_for_test, labels_for_test),
        batch_size=config.batch_size,
        collate_fn=TokenizerWrapper(tokenizer,
                                    config.max_length,
                                    config.entity_max_length).generate_for_train,
    )

    return train_loader, valid_loader, index_to_label, test_loader


def main(config):
    # Get pretrained tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)

    # Get dataloaders using tokenizer from untokenized corpus.
    train_loader, valid_loader, index_to_label, test_loader = get_loaders(config, tokenizer)

    print(
        '|train| =', len(train_loader) * config.batch_size,
        '|valid| =', len(valid_loader) * config.batch_size,
        '|test| =', len(test_loader) * config.batch_size,
    )

    print(index_to_label)

    # Get model
    model = BERT_BiLSTM(
        config,
        n_classes=len(index_to_label),
    )

    if config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    else:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.lr,
            eps=config.adam_epsilon
        )

    n_total_iterations = len(train_loader) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations
    )

    if config.gpu_id > -1 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.gpu_id))
        print('GPU on')
        print('Count of using GPUs:', torch.cuda.device_count())
    else:
        device = torch.device("cpu")
        print('No GPU')

    model.to(device)

    # Start train.
    trainer = Trainer(config)

    model = trainer.train(
        model,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        index_to_label,
        device,
    )

    trainer.test(
        model,
        test_loader,
        index_to_label,
        device,
    )

    torch.save({
        'bert': model.state_dict(),
        'config': config,
        'vocab': tokenizer.get_vocab(),
        'index_to_label': index_to_label,
        'tokenizer': tokenizer,
    }, config.model_fn)


if __name__ == '__main__':

    config = define_argparser()
    main(config)
