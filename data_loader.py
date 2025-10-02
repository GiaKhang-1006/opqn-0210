import torch
import torchvision.transforms as transforms
from torchvision import datasets
import os

def get_datasets_transform(dataset, data_dir="./data", cross_eval=False):
    to_tensor = transforms.ToTensor()
    # Auto detect Kaggle path
    root = '/kaggle/working/opqn-fork/data/' if 'kaggle' in os.environ.get('PWD', '') else data_dir

    # Define paths with folder existence check
    if dataset != "vggface2":
        train_path = os.path.join(root, dataset, "train")
        test_path = os.path.join(root, dataset, "test")
        if not os.path.exists(train_path):
            train_path = os.path.join(root, dataset)  # Fallback nếu không có split
        if not os.path.exists(test_path):
            test_path = os.path.join(root, dataset)  # Fallback
    else:
        if cross_eval:  # vggface2 cross-dataset retrieval
            train_path = os.path.join(root, "vggface2", "cross_train")
            test_path = os.path.join(root, "vggface2", "cross_test")
            if not os.path.exists(train_path):
                train_path = os.path.join(root, "vggface2", "train")
                print("Warning: cross_train not found, using train instead")
            if not os.path.exists(test_path):
                test_path = os.path.join(root, "vggface2", "test")
                print("Warning: cross_test not found, using test instead")
        else:
            train_path = os.path.join(root, "vggface2", "train")
            test_path = os.path.join(root, "vggface2", "test")

    # Load datasets
    trainset = datasets.ImageFolder(root=train_path, transform=to_tensor)
    testset = datasets.ImageFolder(root=test_path, transform=to_tensor)

    # Define transforms
    if cross_eval:
        transform_train = torch.nn.Sequential(
            transforms.Resize(120),
            transforms.CenterCrop(112),
            # transforms.RandomHorizontalFlip(),  # Comment để giữ nhất quán với test
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        )
        transform_test = transform_train
    else:
        if dataset == "vggface2":  # Sửa lỗi chính tả datasets -> dataset
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

    # Debug print
    print(f"Dataset: {dataset}, Cross-eval: {cross_eval}")
    print(f"Train path: {train_path}, Test path: {test_path}")

    return {"dataset": [trainset, testset], "transform": [transform_train, transform_test]}