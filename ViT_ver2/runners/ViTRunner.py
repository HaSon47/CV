import torch
from transformers import ViTForImageClassification
from torch import optim, nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from model.ViT import ViT
from utils.utils import check_device, split_data
class ViTRunner:
    def __init__(self, config):
        
        #self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.epochs = config.runner_model.epochs
        self.lr = config.runner_model.lr
        self.device = check_device()
        self.num_classes = config.ViT.out_dim

        self.chechpoint_path = config.runner_model.chechpoint_path
        self.log_interval = config.runner_model.log_interval
        self.metrics = None

        #history[loss,accuracy]
        self.train_history = []
        self.val_history = []

        self.use_pretrained = config.runner_model.use_pretrained
        if self.use_pretrained:
            print('pretrained_ViT')
            self.model = ViTForImageClassification.from_pretrained(config.runner_model.pretrained_model_path)
            # frozen
            for param in self.model.vit.parameters():
                param.requires_grad = False
                self.model.classifier = nn.Linear(in_features=768, out_features=self.num_classes, bias=True)
        else:
            print('ViT from scrach')
            self.model = ViT(config, self.device)
        self.model.to(self.device)

    def init_data(self, dataset, spilt_ratio=[0.8,0.1,0.1], batch_size=8):
        self.train_loader, self.test_loader, self.val_loader = split_data(dataset,spilt_ratio,batch_size)

    def train(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9)
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            loop = tqdm(self.train_loader, desc=f'Epoch{epoch+1}/{self.epochs}')
            for batch in loop:
                images, labels = batch['pixel_values'].to(self.device), batch['labels'].to(self.device)

                #Forward
                outputs = self.model(images)
                if self.use_pretrained:
                    outputs = outputs.logits #####

                #loss
                loss = self.loss_fn(outputs, labels)

                #Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                #Tính loss và accuracy
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                loop.set_postfix(loss=loss.item(), accuracy=100.*correct/total)
            # Lưu loss và accuracy cho epoch
            train_loss = epoch_loss / len(self.train_loader)
            train_acc = 100. * correct / total  
            self.train_history.append([train_loss,train_acc])
            print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')

            # Chạy validation sau mỗi epoch
            val_loss, val_acc = self.val()
            self.val_history.append([val_loss,val_acc])

    def val(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images, labels = batch['pixel_values'].to(self.device), batch['labels'].to(self.device)

                outputs = self.model(images)
                if self.use_pretrained:
                    outputs = outputs.logits ####

                loss = self.loss_fn(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        val_loss /= len(self.val_loader)
        val_acc = 100.*correct/total
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
        return val_loss, val_acc


    def test(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.test_loader:
                images, labels = batch['pixel_values'].to(self.device), batch['labels'].to(self.device)

                outputs = self.model(images)
                if self.use_pretrained:
                    outputs = outputs.logits ####

                loss = self.loss_fn(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        test_loss /= len(self.test_loader)
        test_acc = 100.*correct/total
        print(f'Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_acc:.2f}%')
        return test_loss, test_acc
    
    def predict(self, image):
        '''predict label for an image'''
        self.model.eval
        with torch.no_grad():
            image = image.to(self.device).unsqueeze(0) # add batch dimmension
            output = self.model(image)
            if self.use_pretrained:
                output = output.logits ####
            _, predicted = output.max(1)
        return predicted.item()

    def plot_history(self):
        epochs_range = range(1,self.epochs+1)
        plt.figure(figsize=(12,5))
        # loss
        plt.subplot(1,2,1)
        plt.plot(epochs_range, [i[0] for i in self.train_history], label='Train loss')
        plt.plot(epochs_range, [i[0] for i in self.val_history], label='Val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss History')
        plt.legend()

        # accuracy
        plt.subplot(1,2,2)
        plt.plot(epochs_range, [i[1] for i in self.train_history], label='Train acc')
        plt.plot(epochs_range, [i[1] for i in self.val_history], label='Val acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy History')
        plt.legend()

        plt.show()

    def save_checkpoint(self, path):
        # Lưu trạng thái của model
        pass
    def load_checkpoint(self, path):
        # Load checkpoint từ file
        pass
    def evaluate_metrics(self, loader):
        # Tính accuracy, F1-score
        pass
