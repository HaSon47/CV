import yaml
from runs.pretrained_ViT import pretrained_ViT
from utils import dict_to_namespace
from dataset import CustomPetDataset
def main():
    config_path = './config.yml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config = dict_to_namespace(config)

    dataset = CustomPetDataset(config.model.path)
    runner = pretrained_ViT(config)
    runner.train(dataset)

if __name__ == "__main__":
    main()

# git.ignore