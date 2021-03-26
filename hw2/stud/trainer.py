import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(
            self,
            model: nn.Module,
            loss_function_s3,
            loss_function_s4,
            optimizer):

        self.model = model
        self.loss_function_s3 = loss_function_s3
        self.loss_function_s4 = loss_function_s4
        self.optimizer = optimizer

    def train(self, train_dataset,
              valid_dataset,
              epochs):

        train_loss = 0.0
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}')

            epoch_loss = 0.0
            self.model.train()

            for step, sentence in enumerate(train_dataset):

                # Create proper format for input sentences
                tokens = {
                    'lemmas': sentence['lemmas'],
                    'pos_tags': sentence['pos_tags'],
                    'predicates': sentence['predicate'],
                    'predicate_indicator': sentence['predicate_indicator'],
                    'lemmas_indicator': sentence['lemmas_indicator']
                }
                labels = {'roles': sentence['roles'],
                          'bi_roles': sentence['bi_roles']}

                self.optimizer.zero_grad()

                # Model will return two loss
                # The first one is for binary labels
                # The second one is for multi labels  
                predictions_s3, predictions_s4 = self.model(tokens)

                predictions_s3 = torch.transpose(predictions_s3, 1, 2)
                predictions_s4 = torch.transpose(predictions_s4, 1, 2)

                temp_loss_s3 = self.loss_function_s3(predictions_s3, labels['bi_roles'])
                temp_loss_s4 = self.loss_function_s4(predictions_s4, labels['roles'])

                # Set loss for '_' tokens to zero
                # So during backward, the model focuses on non -_- items
                temp_loss_s4 = temp_loss_s4 * labels['bi_roles']

                # Cross Entropy reduction is none, so it won't perform mean operation of loss values
                # We needed the raw loss values for above operation
                # So we have to perform mean operation manually
                temp_loss_s3 = temp_loss_s3.mean(dim=-1).mean()
                temp_loss_s4 = temp_loss_s4.mean(dim=-1).mean()

                temp_loss = temp_loss_s3 + temp_loss_s4

                temp_loss.backward()
                self.optimizer.step()

                epoch_loss += temp_loss.tolist()

            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss
            print(f'\t[Epoch: {epoch + 1}] Training Loss = {avg_epoch_loss}')

            valid_loss = self.evaluate(valid_dataset)
            print(f'\t[Epoch: {epoch + 1}] Validation Loss = {valid_loss}')

        print('Training has finished')

        avg_epoch_loss = train_loss / epochs
        return avg_epoch_loss

    def evaluate(self, valid_dataset):

        valid_loss = 0.0
        self.model.eval()

        with torch.no_grad():
            for sentence in valid_dataset:
                tokens = {
                    'lemmas': sentence['lemmas'],
                    'pos_tags': sentence['pos_tags'],
                    'predicates': sentence['predicate'],
                    'predicate_indicator': sentence['predicate_indicator'],
                    'lemmas_indicator': sentence['lemmas_indicator']

                }
                labels = {'roles': sentence['roles'],
                          'bi_roles': sentence['bi_roles']}

                predictions_s3, predictions_s4 = self.model(tokens)

                predictions_s3 = torch.transpose(predictions_s3, 1, 2)
                predictions_s4 = torch.transpose(predictions_s4, 1, 2)

                temp_loss_s3 = self.loss_function_s3(predictions_s3, labels['bi_roles'])
                temp_loss_s4 = self.loss_function_s4(predictions_s4, labels['roles'])


                # Set loss for '_' tokens to zero
                # So during backward, the model focuses on non -_- items
                temp_loss_s4 = temp_loss_s4 * labels['bi_roles']

                temp_loss_s3 = temp_loss_s3.mean(dim=-1).mean()
                temp_loss_s4 = temp_loss_s4.mean(dim=-1).mean()

                temp_loss = temp_loss_s3 + temp_loss_s4

                valid_loss += temp_loss.tolist()

        return valid_loss / len(valid_dataset)

    def predict(self, x):

        self.model.eval()

        with torch.no_grad():
            logits_s3, logits_s4 = self.model(x)
            predictions_s3 = torch.argmax(logits_s3, -1)
            predictions_s4 = torch.argmax(logits_s4, -1)
            predictions = predictions_s3 * predictions_s4
            return predictions
