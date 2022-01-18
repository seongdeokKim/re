import torch
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize


class BertDataset(Dataset):

    def __init__(self, sentences: 'list of str', entity_info: 'list of item(dict)', labels: 'list of str'):
        self.sentences = sentences
        self.entity_info = entity_info
        self.labels = labels
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        entity_info_per_sent = self.entity_info[item]
        label = self.labels[item]

        return sentence, entity_info_per_sent, label

        
class TokenizerWrapper():

    def __init__(self, tokenizer,
                 max_length,
                 entity_max_length):

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.entity_max_length = entity_max_length

        self.CLS, self.CLS_id = self.tokenizer.cls_token, self.tokenizer.cls_token_id
        self.SEP, self.SEP_id = self.tokenizer.sep_token, self.tokenizer.sep_token_id
        self.PAD, self.PAD_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id

    def generate_for_train(self, samples):

        sentences = [s[0] for s in samples]
        entity_info = [s[1] for s in samples]
        labels = [s[2] for s in samples]

        transformed_inputs = [
            self.generate_re_sent(sentence, entity_info_per_sent)
            for sentence, entity_info_per_sent in zip(sentences, entity_info)
        ]

        # Convert tokens to token_ids
        input_ids = [
            self.tokenizer.convert_tokens_to_ids(transformed_inputs_per_sent)
            for transformed_inputs_per_sent in transformed_inputs
        ]

        # Place a mask (zero) over the padding tokens
        attention_mask = [
            [float(input_id > 0) for input_id in input_ids_per_sent]
            for input_ids_per_sent in input_ids
        ]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }

    def generate_for_predict(self, samples: 'list of tuple(str, dict)'):

        sentences = [s[0] for s in samples]
        entity_info = [s[1] for s in samples]

        transformed_inputs = [
            self.generate_re_sent(sentence, entity_info_per_sent)
            for sentence, entity_info_per_sent in zip(sentences, entity_info)
        ]

        # Convert tokens to token_ids
        input_ids = [
            self.tokenizer.convert_tokens_to_ids(transformed_inputs_per_sent)
            for transformed_inputs_per_sent in transformed_inputs
        ]

        # Place a mask (zero) over the padding tokens
        attention_mask = [
            [float(input_id > 0) for input_id in input_ids_per_sent]
            for input_ids_per_sent in input_ids
        ]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }

    def generate_re_sent(self, sent: str, entity_info_per_sent: dict):

        new_sent = []
        left_entity_span = entity_info_per_sent.get("left_entity_span")
        left_entity_span = [int(span.strip()) for span in left_entity_span.split("_")]

        right_entity_span = entity_info_per_sent.get("right_entity_span")
        right_entity_span = [int(span.strip()) for span in right_entity_span.split("_")]

        left_text = sent[:left_entity_span[0]]
        left_tag = entity_info_per_sent.get("left_entity_tag")
        middle_text = sent[left_entity_span[1]:right_entity_span[0]]
        right_tag = entity_info_per_sent.get("right_entity_tag")
        right_text = sent[right_entity_span[1]:]

        new_sent.append(left_text)
        new_sent.append(left_tag)
        new_sent.append(middle_text)
        new_sent.append(right_tag)
        new_sent.append(right_text)
        new_sent = "".join(new_sent)

        tokenized_sent = self.tokenizer.tokenize(new_sent)

        sent_max_len = self.max_length - self.entity_max_length
        if len(tokenized_sent) < sent_max_len:
            sent_pad_len = sent_max_len - len(tokenized_sent)
            tokenized_sent = tokenized_sent + [self.PAD] * (sent_pad_len-1)
        else:
            tokenized_sent = tokenized_sent[:sent_max_len - 1]

        left_entity, right_entity = [], []
        left_entity += self.tokenizer.tokenize(entity_info_per_sent.get("left_entity"))
        right_entity += self.tokenizer.tokenize(entity_info_per_sent.get("right_entity"))

        entity_half_length = self.entity_max_length // 2
        if len(left_entity) < entity_half_length:
            entity_pad_len = entity_half_length - len(left_entity)
            left_entity = left_entity + [self.PAD] * (entity_pad_len-1)
        else:
            left_entity = left_entity[:entity_half_length - 1]

        if len(right_entity) < entity_half_length:
            entity_pad_len = entity_half_length - len(right_entity)
            right_entity = right_entity + [self.PAD] * (entity_pad_len-1)
        else:
            right_entity = right_entity[:entity_half_length - 1]

        re_sent = tokenized_sent + [self.SEP] + left_entity + [self.SEP] + right_entity + [self.SEP]
        #print(re_sent)

        return re_sent

