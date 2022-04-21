import os
import torch
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_dataset(args):
    return globals()[args.dataset](args.data_dir, args.test_domain_index)


def get_domains(dataset):
    return globals()[dataset].DOMAINS


def get_classes(dataset):
    return globals()[dataset].NUM_CLASSES


class ImageLabelDataset(torch.utils.data.Dataset):

    def __init__(self, file_path, images_dir, image_transforms=None):
        self._read_file(file_path)
        self.images_dir = images_dir
        self.image_transforms = image_transforms

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.images_paths[index])
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        label = self.labels[index]

        if self.image_transforms is not None:
            image = self.image_transforms(image)

        return image, label

    def _read_file(self, file_path):
        self.images_paths = []
        self.labels = []
        with open(file_path, 'r') as f:
            for line in f:
                path, label = line.strip().split(',')
                self.images_paths.append(path)
                self.labels.append(int(label) - 1)


class MultiDomainDataset:

    def __init__(self, data_dir, test_domain_index):
        images_dir = os.path.join(data_dir, 'images')
        split_dir = os.path.join(data_dir, 'split')

        domain_names = [f.name for f in os.scandir(images_dir) if f.is_dir()]
        domain_names.sort()

        test_domain = domain_names[test_domain_index]
        train_domains = [d for d in domain_names if d != test_domain]

        image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create train, validation and test datasets
        train_datasets, val_datasets = [], []
        for domain_name in train_domains:
            train_datasets.append(ImageLabelDataset(
                os.path.join(split_dir, domain_name + '_train.txt'),
                images_dir,
                image_transforms)
            )
            val_datasets.append(ImageLabelDataset(
                os.path.join(split_dir, domain_name + '_val.txt'),
                images_dir,
                image_transforms)
            )

        self.datasets = {
            'train': train_datasets,
            'val': torch.utils.data.ConcatDataset(val_datasets),
            'test': ImageLabelDataset(
                os.path.join(split_dir, test_domain + '_test.txt'),
                images_dir,
                image_transforms
            )
        }

    def __getitem__(self, mode):
        if mode in ['train', 'val', 'test']:
            return self.datasets[mode]
        raise ValueError


class PACS(MultiDomainDataset):

    NUM_CLASSES = 7
    DOMAINS = ['A', 'C', 'P', 'S']

    def __init__(self, data_dir, test_domain_index):
        self.data_dir = os.path.join(data_dir, 'PACS/')
        super().__init__(self.data_dir, test_domain_index)


class VLCS(MultiDomainDataset):

    NUM_CLASSES = 5
    DOMAINS = ['C', 'L', 'S', 'V']

    def __init__(self, data_dir, test_domain_index):
        self.data_dir = os.path.join(data_dir, 'VLCS/')
        super().__init__(self.data_dir, test_domain_index)


class OfficeHome(MultiDomainDataset):

    NUM_CLASSES = 65
    DOMAINS = ['A', 'C', 'P', 'R']

    def __init__(self, data_dir, test_domain_index):
        self.data_dir = os.path.join(data_dir, 'OfficeHome/')
        super().__init__(self.data_dir, test_domain_index)
