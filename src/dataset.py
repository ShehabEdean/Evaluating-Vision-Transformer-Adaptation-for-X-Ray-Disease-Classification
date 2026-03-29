import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Pleural_Thickening"
]


def load_label_dict(csv_file: str, labels_filter: list = None) -> dict:
    labels_filter = labels_filter or DISEASE_LABELS
    df = pd.read_csv(csv_file)
    label_dict = {}
    for _, row in df.iterrows():
        img_name = row['Image Index']
        findings = row['Finding Labels'].split('|')
        if findings == ['No Finding'] or findings[0] == 'No Finding':
            vector = [0.0] * len(labels_filter)
        else:
            vector = [1.0 if disease in findings else 0.0 for disease in labels_filter]
        label_dict[img_name] = np.array(vector, dtype=np.float32)
    return label_dict


def scan_images(image_dirs) -> dict:
    path_cache = {}
    for img_dir in image_dirs:
        for img_path in glob.glob(os.path.join(img_dir, "*.png")):
            img_name = os.path.basename(img_path)
            path_cache[img_name] = img_path
    return path_cache


def load_label_dict_with_patient(csv_file: str, labels_filter: list = None) -> tuple:
    labels_filter = labels_filter or DISEASE_LABELS
    df = pd.read_csv(csv_file)
    label_dict = {}
    patient_ids = {}
    for _, row in df.iterrows():
        img_name = row['Image Index']
        patient_ids[img_name] = row['Patient ID']
        findings = row['Finding Labels'].split('|')
        if findings == ['No Finding'] or findings[0] == 'No Finding':
            vector = [0.0] * len(labels_filter)
        else:
            vector = [1.0 if disease in findings else 0.0 for disease in labels_filter]
        label_dict[img_name] = np.array(vector, dtype=np.float32)
    return label_dict, patient_ids


def split_by_patient(csv_file: str, train_ratio: float = 0.8, seed: int = 42) -> tuple:
    df = pd.read_csv(csv_file)
    unique_patients = df['Patient ID'].unique()
    
    np.random.seed(seed)
    np.random.shuffle(unique_patients)
    
    num_train = int(len(unique_patients) * train_ratio)
    train_patients = set(unique_patients[:num_train])
    val_patients = set(unique_patients[num_train:])
    
    train_files = df[df['Patient ID'].isin(train_patients)]['Image Index'].tolist()
    val_files = df[df['Patient ID'].isin(val_patients)]['Image Index'].tolist()
    
    print(f"Patient-based split: {len(train_patients)} train patients, {len(val_patients)} val patients")
    print(f"Image split: {len(train_files)} train images, {len(val_files)} val images")
    
    return train_files, val_files


def validate_alignment(label_dict: dict, path_cache: dict) -> tuple:
    label_files = set(label_dict.keys())
    image_files = set(path_cache.keys())
    
    missing_images = label_files - image_files
    missing_labels = image_files - label_files
    
    if missing_images:
        print(f"WARNING: {len(missing_images)} labels without images (first 5: {list(missing_images)[:5]})")
    if missing_labels:
        print(f"WARNING: {len(missing_labels)} images without labels (first 5: {list(missing_labels)[:5]})")
    
    common_files = label_files & image_files
    print(f"Aligned samples: {len(common_files)}")
    
    return common_files, missing_images, missing_labels


class ChestXRayDataset(Dataset):
    def __init__(self, csv_file: str, image_dirs, transform=None, labels_filter: list = None, validate: bool = True, sample_filter: list = None):
        self.image_dirs = image_dirs if isinstance(image_dirs, list) else [image_dirs]
        self.transform = transform
        self.labels_filter = labels_filter or DISEASE_LABELS
        
        self.label_dict = load_label_dict(csv_file, self.labels_filter)
        self.path_cache = scan_images(self.image_dirs)
        
        if validate:
            common_files, _, _ = validate_alignment(self.label_dict, self.path_cache)
        else:
            common_files = set(self.label_dict.keys()) & set(self.path_cache.keys())
        
        if sample_filter:
            common_files = common_files & set(sample_filter)
        
        self.samples = sorted(list(common_files))
        
        if len(self.samples) == 0:
            raise ValueError("No aligned samples found between labels and images!")
    
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_name = self.samples[idx]
        img_path = self.path_cache[img_name]
        labels = self.label_dict[img_name]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels

    def get_label_distribution(self):
        total_labels = np.array([self.label_dict[img] for img in self.samples])
        counts = total_labels.sum(axis=0)
        for disease, count in zip(self.labels_filter, counts):
            pct = count / len(self) * 100
            print(f"{disease}: {int(count):5d} ({pct:.2f}%)")
        no_finding = int((total_labels.sum(axis=1) == 0).sum())
        print(f"\n'No Finding' (all zeros): {no_finding} samples")
