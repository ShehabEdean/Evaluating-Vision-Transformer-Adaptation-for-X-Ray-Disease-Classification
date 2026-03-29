from torchvision import transforms


def grayscale_to_rgb(img):
    if img.mode == 'L':
        img = img.convert('RGB')
    return img


def get_train_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Lambda(grayscale_to_rgb),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Lambda(grayscale_to_rgb),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
