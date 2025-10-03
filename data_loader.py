# import torch
# import torchvision.transforms as transforms
# from torchvision import datasets
# import os
# from torch.utils.data import ConcatDataset

# def get_datasets_transform(dataset, data_dir="/kaggle/input/facescrub-0210-2", cross_eval=False):
#     to_tensor = transforms.ToTensor()

#     # Define paths for FaceScrub processed data
#     if dataset == "facescrub":
#         # Kết hợp cả actor và actress cho train và test
#         train_paths = [
#             os.path.join(data_dir, "facescrub", "train", "actor"),
#             os.path.join(data_dir, "facescrub", "train", "actress")
#         ]
#         test_paths = [
#             os.path.join(data_dir, "facescrub", "test", "actor"),
#             os.path.join(data_dir, "facescrub", "test", "actress")
#         ]
#     elif dataset == "vggface2":
#         if cross_eval:
#             train_path = os.path.join(data_dir, "vggface2", "cross_train")
#             test_path = os.path.join(data_dir, "vggface2", "cross_test")
#         else:
#             train_path = os.path.join(data_dir, "vggface2", "train")
#             test_path = os.path.join(data_dir, "vggface2", "test")
#     else:
#         train_path = os.path.join(data_dir, dataset, "train")
#         test_path = os.path.join(data_dir, dataset, "test")

#     # Load datasets with debug print
#     if dataset == "facescrub":
#         trainsets = [datasets.ImageFolder(root=path, transform=to_tensor) for path in train_paths]
#         testsets = [datasets.ImageFolder(root=path, transform=to_tensor) for path in test_paths]
#         trainset = ConcatDataset(trainsets)
#         testset = ConcatDataset(testsets)
#     else:
#         trainset = datasets.ImageFolder(root=train_path, transform=to_tensor)
#         testset = datasets.ImageFolder(root=test_path, transform=to_tensor)

#     print(f"Train paths: {train_paths}")  # Debug
#     print(f"Test paths: {test_paths}")    # Debug
#     print(f"Train classes: {trainset.classes}")  # Debug: Print number of identities
#     print(f"Test classes: {testset.classes}")    # Debug: Print number of identities

#     if cross_eval:
#         transform_train = torch.nn.Sequential(
#             transforms.Resize(120),
#             transforms.CenterCrop(112),
#             transforms.ConvertImageDtype(torch.float),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         )
#         transform_test = transform_train
#     else:
#         if dataset == "vggface2":
#             transform_train = torch.nn.Sequential(
#                 transforms.Resize(120),
#                 transforms.RandomCrop(112),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             )
#             transform_test = torch.nn.Sequential(
#                 transforms.Resize(120),
#                 transforms.CenterCrop(112),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             )
#         else:
#             transform_train = torch.nn.Sequential(
#                 transforms.Resize(35),
#                 transforms.RandomCrop(32),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
#             )
#             transform_test = torch.nn.Sequential(
#                 transforms.Resize(35),
#                 transforms.CenterCrop(32),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
#             )

#     # Áp dụng transform sau khi ghép dataset
#     trainset.transform = transform_train
#     testset.transform = transform_test

#     return {"dataset": [trainset, testset], "transform": [transform_train, transform_test]}











import torch
import torchvision.transforms as transforms
from torchvision import datasets
import os

def get_datasets_transform(dataset, data_dir="/kaggle/input/facescrub-0210-3", cross_eval=False):
    to_tensor = transforms.ToTensor()

    # Define paths for FaceScrub processed data
    if dataset == "facescrub":
        train_path = os.path.join(data_dir, "facescrub", "train", "actors")
        test_path = os.path.join(data_dir, "facescrub", "test", "actors")
    elif dataset == "vggface2":
        if cross_eval:
            train_path = os.path.join(data_dir, "vggface2", "cross_train")
            test_path = os.path.join(data_dir, "vggface2", "cross_test")
        else:
            train_path = os.path.join(data_dir, "vggface2", "train")
            test_path = os.path.join(data_dir, "vggface2", "test")
    else:
        train_path = os.path.join(data_dir, dataset, "train")
        test_path = os.path.join(data_dir, dataset, "test")

    # Load datasets with debug print
    trainset = datasets.ImageFolder(root=train_path, transform=to_tensor)
    testset = datasets.ImageFolder(root=test_path, transform=to_tensor)
    print(f"Train path: {train_path}")  # Debug
    print(f"Test path: {test_path}")    # Debug
    print(f"Train classes: {trainset.classes}")  # Debug: Print number of identities
    print(f"Test classes: {testset.classes}")    # Debug: Print number of identities

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










# import torch
# import torchvision.transforms as transforms
# from torchvision import datasets

# import os

# def get_datasets_transform(dataset, data_dir="/kaggle/input/facescrub-0210-1", cross_eval=False):
#     to_tensor = transforms.ToTensor()
#     if dataset!="vggface2":
#         trainPaths = os.path.join(data_dir, dataset, "train") 
#         testPaths = os.path.join(data_dir, dataset, "test")
#     else:
#         if cross_eval: # vgggface2 cross-dataset retrieval uses another train-test splits from standard retrieval
#             trainPaths = os.path.join(data_dir, "vggface2", "cross_train") 
#             testPaths = os.path.join(data_dir, "vggface2", "cross_test")
#         else:
#             trainPaths = os.path.join(data_dir, "vggface2", "train") 
#             testPaths = os.path.join(data_dir, "vggface2", "test")
#     trainset = datasets.ImageFolder(root=trainPaths, transform=to_tensor)
#     testset = datasets.ImageFolder(root=testPaths, transform=to_tensor)
#     if cross_eval:
#         transform_train = torch.nn.Sequential(
#                     transforms.Resize(120),
#                     transforms.CenterCrop(112),
#                     # transforms.RandomHorizontalFlip(),
#                     transforms.ConvertImageDtype(torch.float),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         )

#         transform_test = transform_train

#     else:
#         if dataset=="vggface2":
#             transform_train = torch.nn.Sequential(
#                     transforms.Resize(120),
#                     transforms.RandomCrop(112),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ConvertImageDtype(torch.float),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             )

#             transform_test = torch.nn.Sequential(
#                     transforms.Resize(120),
#                     transforms.CenterCrop(112),
#                     transforms.ConvertImageDtype(torch.float),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             )
        
#         else:
#             transform_train = torch.nn.Sequential(
#                     transforms.Resize(35), 
#                     transforms.RandomCrop(32),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ConvertImageDtype(torch.float),
#                     transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
#             )

#             transform_test = torch.nn.Sequential(
#                     transforms.Resize(35), 
#                     transforms.CenterCrop(32),
#                     transforms.ConvertImageDtype(torch.float),
#                     transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
#             )
#     return {"dataset": [trainset, testset], "transform": [transform_train, transform_test]}
    





# import torch
# import torchvision.transforms as transforms
# from torchvision import datasets
# import os

# def get_datasets_transform(dataset, data_dir="./data", cross_eval=False):
#     to_tensor = transforms.ToTensor()

#     # Auto detect Kaggle and use /kaggle/input/ processed paths
#     if 'kaggle' in os.environ.get('PWD', ''):
#         if dataset == 'facescrub':
#             base_path = '/kaggle/input/processed_facescrub/processed_facescrub/'  # Đường dẫn đúng với dataset của bạn
#         else:
#             base_path = data_dir  # Fallback cho dataset khác
#     else:
#         base_path = data_dir  # Cục bộ

#     # Define paths with folder existence check
#     train_path = os.path.join(base_path, "train")
#     test_path = os.path.join(base_path, "test")
#     if not os.path.exists(train_path):
#         train_path = base_path  # Fallback nếu không có split
#     if not os.path.exists(test_path):
#         test_path = train_path  # Fallback

#     # Debug print
#     print(f"Dataset: {dataset}, Cross-eval: {cross_eval}")
#     print(f"Train path: {train_path}, Test path: {test_path}")

#     trainset = datasets.ImageFolder(root=train_path, transform=to_tensor)
#     testset = datasets.ImageFolder(root=test_path, transform=to_tensor)

#     # Transforms
#     if cross_eval:
#         transform_train = torch.nn.Sequential(
#             transforms.Resize(120),
#             transforms.CenterCrop(112),
#             transforms.ConvertImageDtype(torch.float),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         )
#         transform_test = transform_train
#     else:
#         if dataset == "vggface2":
#             transform_train = torch.nn.Sequential(
#                 transforms.Resize(120),
#                 transforms.RandomCrop(112),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             )
#             transform_test = torch.nn.Sequential(
#                 transforms.Resize(120),
#                 transforms.CenterCrop(112),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             )
#         else:
#             transform_train = torch.nn.Sequential(
#                 transforms.Resize(35),
#                 transforms.RandomCrop(32),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
#             )
#             transform_test = torch.nn.Sequential(
#                 transforms.Resize(35),
#                 transforms.CenterCrop(32),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
#             )

#     return {"dataset": [trainset, testset], "transform": [transform_train, transform_test]}




# import torch
# import torchvision.transforms as transforms
# from torchvision import datasets
# import os

# def get_datasets_transform(dataset, data_dir="./data", cross_eval=False):
#     to_tensor = transforms.ToTensor()

#     # Auto detect Kaggle and use /kaggle/input/ processed paths
#     if 'kaggle' in os.environ.get('PWD', ''):
#         if dataset == 'facescrub':
#             base_path = '/kaggle/input/processed-facescrub/processed-facescrub/'  # Thay bằng tên dataset bạn upload
#         else:
#             base_path = data_dir  # Fallback cho dataset khác
#     else:
#         base_path = data_dir  # Cục bộ

#     # Define paths with folder existence check
#     if dataset != "vggface2":
#         train_path = os.path.join(base_path, "train")  # Thay vì dataset/train
#         test_path = os.path.join(base_path, "test")    # Thay vì dataset/test
#         if not os.path.exists(train_path):
#             train_path = os.path.join(base_path)  # Fallback nếu không có split
#         if not os.path.exists(test_path):
#             test_path = train_path  # Fallback
#     else:
#         if cross_eval:  # vggface2 cross-dataset
#             train_path = os.path.join(base_path, "cross_train") if os.path.exists(os.path.join(base_path, "cross_train")) else os.path.join(base_path, "train")
#             test_path = os.path.join(base_path, "cross_test") if os.path.exists(os.path.join(base_path, "cross_test")) else os.path.join(base_path, "test")
#         else:
#             train_path = os.path.join(base_path, "train")
#             test_path = os.path.join(base_path, "test")

#     # Debug print
#     print(f"Dataset: {dataset}, Cross-eval: {cross_eval}")
#     print(f"Train path: {train_path}, Test path: {test_path}")

#     trainset = datasets.ImageFolder(root=train_path, transform=to_tensor)
#     testset = datasets.ImageFolder(root=test_path, transform=to_tensor)

#     # Transforms (giữ nguyên)
#     if cross_eval:
#         transform_train = torch.nn.Sequential(
#             transforms.Resize(120),
#             transforms.CenterCrop(112),
#             transforms.ConvertImageDtype(torch.float),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         )
#         transform_test = transform_train
#     else:
#         if dataset == "vggface2":
#             transform_train = torch.nn.Sequential(
#                 transforms.Resize(120),
#                 transforms.RandomCrop(112),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             )
#             transform_test = torch.nn.Sequential(
#                 transforms.Resize(120),
#                 transforms.CenterCrop(112),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             )
#         else:
#             transform_train = torch.nn.Sequential(
#                 transforms.Resize(35),
#                 transforms.RandomCrop(32),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
#             )
#             transform_test = torch.nn.Sequential(
#                 transforms.Resize(35),
#                 transforms.CenterCrop(32),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
#             )

#     return {"dataset": [trainset, testset], "transform": [transform_train, transform_test]}