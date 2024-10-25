from torch.utils.data import Dataset
from transformers import ViTImageProcessor
from torchvision.datasets import OxfordIIITPet

class CustomPetDataset(Dataset):
    def __init__(self, path,img_size=144,transforms=None):

        self.img_process = ViTImageProcessor.from_pretrained(path)
        self.dataset = OxfordIIITPet(root=".", download=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        image, label = self.dataset[id]
        inputs = self.img_process(image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = label
        return inputs