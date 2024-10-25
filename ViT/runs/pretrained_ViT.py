import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from transformers import ViTForImageClassification
class pretrained_ViT:
    def __init__(self, config):
        self.config = config

        self.batch_size = config.data.batch_size
        self.train_dataloader = None
        self.test_dataloader = None
        self.len_train = 0

        self.model = ViTForImageClassification.from_pretrained(config.model.path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()


    def init_data(self, dataset):
        train_split = int(self.config.data.split * len(dataset))
        train, test = random_split(dataset, [train_split, len(dataset)- train_split])
        self.len_train = len(train)

        self.train_dataloader = DataLoader(train, batch_size=self.batch_size, shuffle = True)
        self.test_dataloader = DataLoader(test, batch_size=self.batch_size, shuffle = False)
    def init_model(self):
        #frozen
        for param in self.model.vit.parameters():
            param.requires_grad = False
        #fit last layer with dataset
        self.model.classifier = nn.Linear(in_features=768, out_features=self.config.data.num_class, bias=True)

        self.model.to(self.device)
    
    def train(self, dataset):
        self.init_data(dataset)
        self.init_model()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.model.lr, momentum=self.config.model.momentum)

        self.model.train()
        for epoch in range(self.config.model.epoch):
            print("--epoch--")
            train_loss = 0.0

            for step, batch in enumerate(self.train_dataloader):
                inputs, labels = batch['pixel_values'].to(self.device), batch['labels'].to(self.device)


                # if step==0:
                #     print(labels[0])
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                outputs = outputs.logits
                # if step==0:
                #     print(outputs.shape)
                #     print(labels.shape)
                loss = self.criterion(outputs, labels)
                loss.backward()

                self.optimizer.step()
                #scheduler.step()

                train_loss += loss.item()*inputs.size(0)

            avg_train_loss = train_loss/self.len_train

            print(f'Epoch {epoch}: train loss: {avg_train_loss:.4f}')


    