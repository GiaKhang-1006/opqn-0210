import torch
import torchvision.transforms as transforms
from torchvision import datasets
import os

def get_datasets_transform(dataset, data_dir="./data", cross_eval=False):
    to_tensor = transforms.ToTensor()

    # Auto detect Kaggle and use /kaggle/input/ paths
    if 'kaggle' in os.environ.get('PWD', ''):
        if dataset == 'facescrub':
            base_path = '/kaggle/input/facescrub-full/'  # Thay bằng tên dataset bạn add (kiểm tra tab Data)
        elif dataset == 'vggface2':
            base_path = '/kaggle/input/vggface2-112x112/'  # Thay bằng tên dataset bạn add
        else:
            base_path = data_dir  # Fallback cho dataset khác
    else:
        base_path = data_dir  # Cục bộ

    # Define paths with folder existence check
    if dataset != "vggface2":
        train_path = os.path.join(base_path, dataset, "train") if os.path.exists(os.path.join(base_path, dataset, "train")) else os.path.join(base_path, dataset)
        test_path = os.path.join(base_path, dataset, "test") if os.path.exists(os.path.join(base_path, dataset, "test")) else train_path
    else:
        if cross_eval:  # vggface2 cross-dataset
            train_path = os.path.join(base_path, "vggface2", "cross_train") if os.path.exists(os.path.join(base_path, "vggface2", "cross_train")) else os.path.join(base_path, "vggface2", "train")
            test_path = os.path.join(base_path, "vggface2", "cross_test") if os.path.exists(os.path.join(base_path, "vggface2", "cross_test")) else os.path.join(base_path, "vggface2", "test")
        else:
            train_path = os.path.join(base_path, "vggface2", "train")
            test_path = os.path.join(base_path, "vggface2", "test")

    # Debug print
    print(f"Dataset: {dataset}, Cross-eval: {cross_eval}")
    print(f"Train path: {train_path}, Test path: {test_path}")

    trainset = datasets.ImageFolder(root=train_path, transform=to_tensor)
    testset = datasets.ImageFolder(root=test_path, transform=to_tensor)

    # Transforms (giữ nguyên)
    if cross_eval:
        transform_train = torch.nn.Sequential(
            transforms.Resize(120),
            transforms.CenterCrop(112),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        )
        transform_test = transform_train
    else:
        if dataset == "vggface2":
            transform_train = torch.nn.Sequential(
                transforms.Resize(120),
                transforms.RandomCrop(112),
                transforms.RandomHorizontalFlip(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            )
            transform_test = torch.nn.Sequential(
                transforms.Resize(120),
                transforms.CenterCrop(112),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            )
        else:
            transform_train = torch.nn.Sequential(
                transforms.Resize(35),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
            )
            transform_test = torch.nn.Sequential(
                transforms.Resize(35),
                transforms.CenterCrop(32),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
            )

    return {"dataset": [trainset, testset], "transform": [transform_train, transform_test]}