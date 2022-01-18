from transformers import BertModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class BERT_BiLSTM(nn.Module):
    def __init__(
            self,
            config,
            n_classes,
            lstm_hidden_size=512,
            n_layers=1,
            bert_dropout_p=0.1,
            bilstm_dropout_p=0.1
    ):
        super(BERT_BiLSTM, self).__init__()
        #super().__init__()

        self.bert_dropout_p = bert_dropout_p
        self.bert = BertModel.from_pretrained(
            config.pretrained_model_name,
        )
        self.bert_dropout = nn.Dropout(bert_dropout_p)
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.lstm_input_size = 768
        self.n_classes = n_classes
        self.lstm_hidden_size = lstm_hidden_size
        self.n_layers = n_layers
        self.bilstm_dropout_p = bilstm_dropout_p
        self.text_max_length = config.max_length - config.entity_max_length

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.n_layers,
            dropout=self.bilstm_dropout_p,
            batch_first=True,
            bidirectional=True,
        )
        self.bilstm_dropout = nn.Dropout(bilstm_dropout_p)
        self.generater = nn.Linear(self.lstm_hidden_size * 2, self.n_classes)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
    ):

        sequence_output, pooled_output = self.bert(
            input_ids,
            attention_mask=attention_mask
        )
        # |sequence_output| = (batch_size, max_seq_length, 768)
        # |pooled_output| = (batch_size, 768)

        sequence_output = self.bert_dropout(sequence_output)
        # |sequence_output| = (batch_size, max_seq_length, 768)

        text = sequence_output[:, :self.text_max_length, :]
        # |text| = (batch_size, text_max_length, 768)

        # https://discuss.pytorch.org/t/get-forward-and-backward-output-seperately-from-bidirectional-rnn/2523/2
        concat_output, _ = self.lstm(text)
        # |concat_output| = (batch_size, text_max_length, lstm_hidden_size * 2)

        concat_output = self.bilstm_dropout(concat_output)
        # |concat_output| = (batch_size, text_length, lstm_hidden_size * 2)

        forward_output = concat_output[:, :, :self.lstm_hidden_size]
        # |forward_output| = (batch_size, text_length, lstm_hidden_size)
        backward_output = concat_output[:, :, self.lstm_hidden_size:]
        # |backward_output| = (batch_size, text_length, lstm_hidden_size)

        concat = torch.cat(
            (forward_output[:, -1, :], backward_output[:, 0, :]), dim=-1
        )
        # |concat| = (batch_size, lstm_hidden_size * 2)

        logits = self.generater(concat)
        # |logits| = (batch_size, n_classes)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.n_classes), labels.view(-1))

        output = (logits,)
        return ((loss,) + output) if loss is not None else output