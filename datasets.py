import os
import torch
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_domains(dataset_name):
    return globals()[dataset_name].DOMAINS


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, file_path, image_dir, transform=None):
        self.file_path = file_path
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self._read_file()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, self.image_paths[idx])

        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def _read_file(self):
        with open(self.file_path) as f:
            for line in f:
                path, label = line.strip().split(',')
                self.image_paths.append(path)
                self.labels.append(int(label) - 1)


class MultiDomainDataset:

    def __init__(self, root_dir, test_dom_idx):
        images_dir = os.path.join(root_dir, 'images')
        split_dir = os.path.join(root_dir, 'split')

        domains = [f.name for f in os.scandir(images_dir) if f.is_dir()]
        domains.sort()

        test_dom = domains[test_dom_idx]
        train_doms = [d for d in domains if d != test_dom]

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])

        train_datasets, val_datasets = [], []
        for dom_name in train_doms:
            train_datasets.append(ImageDataset(
                os.path.join(split_dir, dom_name + '_train.txt'),
                images_dir,
                transform))
            val_datasets.append(ImageDataset(
                os.path.join(split_dir, dom_name + '_val.txt'),
                images_dir,
                transform))

        self.datasets = {}
        self.datasets['train'] = train_datasets
        self.datasets['val'] = torch.utils.data.ConcatDataset(val_datasets)
        self.datasets['test'] = ImageDataset(
            os.path.join(split_dir, test_dom + '_test.txt'),
            images_dir,
            transform)

    def __getitem__(self, phase):
        if phase in ['train', 'val', 'test']:
            return self.datasets[phase]
        else:
            raise ValueError


class VLCS(MultiDomainDataset):

    N_CLASSES = 5
    DOMAINS = ['C', 'L', 'S', 'V']

    def __init__(self, root_dir, test_dom_idx):
        self.root_dir = os.path.join(root_dir, 'VLCS/')
        super().__init__(self.root_dir, test_dom_idx)


class PACS(MultiDomainDataset):

    N_CLASSES = 7
    DOMAINS = ['A', 'C', 'P', 'S']

    def __init__(self, root_dir, test_dom_idx):
        self.root_dir = os.path.join(root_dir, 'PACS/')
        super().__init__(self.root_dir, test_dom_idx)


class OfficeHome(MultiDomainDataset):

    N_CLASSES = 65
    DOMAINS = ['A', 'C', 'P', 'R']

    def __init__(self, root_dir, test_dom_idx):
        self.root_dir = os.path.join(root_dir, 'OfficeHome/')
        super().__init__(self.root_dir, test_dom_idx)
