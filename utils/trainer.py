from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from seqeval.metrics import f1_score, accuracy_score, classification_report


class Trainer():

    def __init__(self, config):
        self.config = config
        self.best_model = None
        self.best_loss = np.inf
        #self.best_accuracy = 0.0
        self.classification_report = None

    def check_best(self, model, performance, report):
        loss = float(performance)
        if loss <= self.best_loss: # If current epoch returns lower validation loss,
            self.best_loss = loss  # Update lowest validation loss.
            self.best_model = deepcopy(model.state_dict()) # Update best model weights.
            self.classification_report = report

        #accuracy = float(performance)
        #if accuracy >= self.best_accuracy:
        #    self.best_accuracy = accuracy
        #    self.best_model = deepcopy(model.state_dict())
        #self.classification_report = report

    def train(self, model,
              optimizer,
              scheduler,
              train_loader,
              valid_loader,
              index_to_label,
              device,):

        for epoch in range(self.config.n_epochs):

            # Put the model into training mode.
            model.train()
            # Reset the total loss for this epoch.
            total_tr_loss = 0

            for step, mini_batch in enumerate(train_loader):
                input_ids, labels = mini_batch['input_ids'], mini_batch['labels']
                input_ids, labels = input_ids.to(device), labels.to(device)
                attention_mask = mini_batch['attention_mask']
                attention_mask = attention_mask.to(device)

                # You have to reset the gradients of all model parameters
                # before to take another step in gradient descent.
                optimizer.zero_grad()

                # Take feed-forward
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss, logits = outputs[0], outputs[1]

                # Perform a backward pass to calculate the gradients.
                loss.backward()
                # track train loss
                total_tr_loss += loss.item()
                # update parameters
                optimizer.step()
                # Update the learning rate.
                scheduler.step()


            # Calculate the average loss over the training data.
            avg_tr_loss = total_tr_loss / len(train_loader)
            print('Epoch {} - loss={:.4e}'.format(
                epoch+1,
                avg_tr_loss
            ))

            # Put the model into evaluation mode
            model.eval()
            # Reset the validation loss and accuracy for this epoch.
            total_val_loss, total_val_accuracy = 0, 0
            preds, true_labels = [], []
            for step, mini_batch in enumerate(valid_loader):
                input_ids, labels = mini_batch['input_ids'], mini_batch['labels']
                input_ids, labels = input_ids.to(device), labels.to(device)
                attention_mask = mini_batch['attention_mask']
                attention_mask = attention_mask.to(device)

                # Telling the model not to compute or store gradients,
                # saving memory and speeding up validation
                with torch.no_grad():
                    outputs = model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss, logits = outputs[0], outputs[1]

                    # Calculate the accuracy for this batch of test sentences.
                    total_val_loss += loss.mean().item()

                    # Calculate accuracy only if 'y' is LongTensor,
                    # which means that 'y' is one-hot representation.
                    if isinstance(labels, torch.LongTensor) or isinstance(labels, torch.cuda.LongTensor):
                        accuracy = (torch.argmax(logits, dim=-1) == labels).sum() / float(labels.size(0))
                    else:
                        accuracy = 0

                    total_val_accuracy += float(accuracy)

                    # Move logits and labels to CPU
                    logits = logits.detach().cpu().numpy()
                    labels = labels.to('cpu').numpy()
                    # 2d array
                    for pred in np.argmax(logits, axis=-1):
                        preds.append([pred])
                    for label in labels:
                        true_labels.append([label])


            pred_classes, true_classes = [], []
            for pred, true_label in zip(preds, true_labels):
                pred_classes.append(
                    [index_to_label.get(pred[0])]
                )
                true_classes.append(
                    [index_to_label.get(true_label[0])]
                )

            avg_val_loss = total_val_loss / len(valid_loader)
            avg_val_acc = total_val_accuracy / len(valid_loader)
            avg_val_f1_score = f1_score(pred_classes, true_classes)
            self.check_best(model, avg_val_loss, classification_report(pred_classes, true_classes))

            #self.check_best(model, avg_val_loss)

            print('Validation - loss={:.4e} accuracy={:.4f} f1-score={:.4f} best_loss={:.4f}'.format(
                avg_val_loss,
                avg_val_acc,
                avg_val_f1_score,
                self.best_loss,
            ))

        print()
        print(self.classification_report)
        model.load_state_dict(self.best_model)

        return model

    def test(self, model,
             test_loader,
             index_to_label,
             device):

        # Put the model into evaluation mode
        model.eval()
        # Reset the validation loss and accuracy for this epoch.
        total_test_loss, total_test_accuracy = 0, 0
        preds, true_labels = [], []
        for step, mini_batch in enumerate(test_loader):
            input_ids, labels = mini_batch['input_ids'], mini_batch['labels']
            input_ids, labels = input_ids.to(device), labels.to(device)
            attention_mask = mini_batch['attention_mask']
            attention_mask = attention_mask.to(device)

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss, logits = outputs[0], outputs[1]

                # Calculate the accuracy for this batch of test sentences.
                total_test_loss += loss.mean().item()

                # Calculate accuracy only if 'y' is LongTensor,
                # which means that 'y' is one-hot representation.
                if isinstance(labels, torch.LongTensor) or isinstance(labels, torch.cuda.LongTensor):
                    accuracy = (torch.argmax(logits, dim=-1) == labels).sum() / float(labels.size(0))
                else:
                    accuracy = 0

                total_test_accuracy += float(accuracy)

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                labels = labels.to('cpu').numpy()
                # 2d array
                for pred in np.argmax(logits, axis=-1):
                    preds.append([pred])
                for label in labels:
                    true_labels.append([label])

        pred_classes, true_classes = [], []
        for pred, true_label in zip(preds, true_labels):
            pred_classes.append(
                [index_to_label.get(pred[0])]
            )
            true_classes.append(
                [index_to_label.get(true_label[0])]
            )

        avg_test_loss = total_test_loss / len(test_loader)
        avg_test_acc = total_test_accuracy / len(test_loader)
        avg_val_f1_score = f1_score(pred_classes, true_classes)

        print('Test - loss={:.4e} accuracy={:.4f} f1-score={:.4f}'.format(
            avg_test_loss,
            avg_test_acc,
            avg_val_f1_score,
        ))
        print()
        print(self.classification_report)

