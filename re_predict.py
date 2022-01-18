import sys
import argparse

import torch
import torch.nn.functional as F

from utils.models.bert_bilstm import BERT_BiLSTM
from utils.data_loader import TokenizerWrapper


def define_argparser():
    '''
    Define argument parser to take inference using fine-tuned model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--predict_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--top_k', type=int, default=1)

    config = p.parse_args()

    return config


def read_file(file):

    with open(file, 'r', encoding='utf-8') as f:
        samples = []
        for line in f:
            field_line = line.strip().split("\t")

            if len(field_line) > 7:
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
                    (sentence, d)
                )

    return samples


def main(config):
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    train_config = saved_data['config']
    bert_best = saved_data['bert']
    index_to_label = saved_data['index_to_label']

    samples = read_file(config.predict_fn)
    with torch.no_grad():
        # Declare model and load pre-trained weights.
        #tokenizer = AutoTokenizer.from_pretrained(train_config.pretrained_model_name)
        loaded_tokenizer = saved_data['tokenizer']
        tokenizer = TokenizerWrapper(loaded_tokenizer,
                                     train_config.max_length,
                                     train_config.entity_max_length)

        model = BERT_BiLSTM(train_config,
                            n_classes=len(index_to_label))

        model.load_state_dict(bert_best)

        if config.gpu_id > -1:
            model.cuda(config.gpu_id)
        device = next(model.parameters()).device

        # Don't forget turn-on evaluation mode.
        model.eval()

        y_hats = []
        for idx in range(0, len(samples), config.batch_size):
            mini_batch = tokenizer.generate_for_predict(samples[idx:idx+config.batch_size])

            input_ids = mini_batch['input_ids']
            input_ids = input_ids.to(device)
            attention_mask = mini_batch['attention_mask']
            attention_mask = attention_mask.to(device)

            # Take feed-forward
            logits = model(input_ids,
                           attention_mask=attention_mask)[0]

#           current_y_hats = np.argmax(logits, axis=-1)
            current_y_hats = F.softmax(logits, dim=-1)
            y_hats += [current_y_hats]

        # Concatenate the mini-batch wise result
        y_hats = torch.cat(y_hats, dim=0)
        probs, indices = y_hats.cpu().topk(k=len(index_to_label))

        for i in range(len(samples)):
            sys.stdout.write('{}\t{}\t{}\t{}\n'.format(
                ",".join([index_to_label.get(int(j)) for j in indices[i][:config.top_k]]),
                ",".join([str(float(j))[:6] for j in probs[i][:config.top_k]]),
                samples[i][0],
                "||".join([k + " : " + v for k, v in samples[i][1].items()]),
            ))


if __name__ == '__main__':

    config = define_argparser()
    main(config)
