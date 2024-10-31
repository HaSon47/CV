import yaml
from runners.ViTRunner import ViTRunner
from utils.utils import dict_to_namespace
from data.OxfordIIIPetDataset import OxfordIIIPetDataset
def main():
    config_path = './config/config.yml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config = dict_to_namespace(config)

    run = ViTRunner(config)
    dataset = OxfordIIIPetDataset(config.dataset.oxfordiiipet_folder_path, config.dataset.model_path, pretrained=config.runner_model.use_pretrained)
    run.init_data(dataset)
    run.train()
    run.test()

if __name__ == "__main__":
    main()
