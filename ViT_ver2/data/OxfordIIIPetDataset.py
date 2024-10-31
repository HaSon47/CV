from torch.utils.data import Dataset
from transformers import ViTImageProcessor
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import Resize, ToTensor, Compose

class OxfordIIIPetDataset(Dataset):  
    def __init__(self, image_root, model_path=None, pretrained=False, download=False, img_size=144):
        self.pretrained = pretrained
        if pretrained:
            self.img_process = ViTImageProcessor.from_pretrained(model_path)
            self.dataset = OxfordIIITPet(root=image_root, download=download)
        else:
            self.transform = Compose([
                Resize((img_size, img_size)),
                ToTensor()
            ])
            self.dataset = OxfordIIITPet(root=image_root, download=download, transform= self.transform )
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, id):
        if self.pretrained:
            image, label = self.dataset[id]
            inputs = self.img_process(image, return_tensors="pt")
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs['labels'] = label
        else:
            image, label = self.dataset[id]
            inputs = {}
            inputs['pixel_values'] = image
            inputs['labels'] = label
        return inputs
    