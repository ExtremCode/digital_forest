import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tifffile import imread
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torchmetrics
from PIL import Image
from os import listdir
from typing import Tuple, List, Dict, Any
import json
from models.SegNet import SegNet
from models.LeNet import LeNet

labels = sorted(['ABBA', 'ACPE', 'ACRU', 'ACSA', 'BEAL', 
                  'BEPA', 'FAGR', 'LALA', 'Mort', 'PIST', 
                  'Picea', 'Populus', 'THOC', 'TSCA'])


def get_dataset_subimage(subimage_id: str,
                         folder="Train",
                         transform=None,
                         need_normalize=False) \
    -> Tuple[torch.Tensor, torch.Tensor]:

    """
    Return a tuple of image tensor and appropriate mask tensor from folder
    Args:
        subimage_id -- string like '1-09-02-0-0'
    """
    subimage = imread(f'{folder}/image/crop-z{subimage_id}.tif')
    subimage = transforms.ToTensor()(subimage)[:3]
    # if transform:
    #     transform(subimage)
    # transform to tensor and remove useless alpha channel
    # subimage = transforms.ToTensor()(pic=subimage)[:3]
    # if need_normalize:
    #     norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     subimage = norm(subimage)

    subimage_labeled = imread(f'{folder}/mask/mask-z{subimage_id}.tif')
    subimage_labeled.resize((264, 264))
    # subimage_labeled = Image.fromarray(obj=subimage_labeled)
    # if transform:
    #     transform(subimage_labeled)
    # subimage_labeled = transforms.ToTensor()(subimage_labeled)
    # subimage_labeled = transforms.Resize((264, 264))(subimage_labeled).round()
    return subimage, subimage_labeled


def get_subimages_proba(subimage_id: str, folder="Train")\
-> Tuple[torch.Tensor, torch.Tensor]:
    
    subimage = imread(f'{folder}/image/crop-z{subimage_id}.tif')
    subimage = transforms.ToTensor()(subimage)[:3]
    with open(f'{folder}/proba/proba-z{subimage_id}.txt', 'r') as file:
        label_proba = torch.tensor(
            [float(el) for el in file.readline().split(" ")],
            dtype=torch.float)

    return subimage, label_proba


def get_image_mask_from_labeled(
        image_labeled: torch.Tensor,
        num_classes: int) -> torch.Tensor:
    """
    Transform mask tensor from (1, 264, 264) to (14, 264, 264)
    i.e. make mask for each label that has size 264*264
    """
    mask = torch.zeros(size=(num_classes,
                    image_labeled.shape[1],
                    image_labeled.shape[2]), dtype=torch.float)
    img = image_labeled[0]
    for r in np.arange(stop=image_labeled.shape[1]):
        for c in np.arange(stop=image_labeled.shape[2]):
            lab = int(img[r][c])
            mask[lab][r][c] = 1.0

    return mask


def get_image_labeled_from_mask(image_mask: np.ndarray) -> Image.Image:
    colors = [
        (0, 0, 0),
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 128, 128),
        (128, 0, 0),
        (0, 128, 0),
        (0, 0, 128),
        (128, 128, 0),
        (0, 128, 128),
        (128, 0, 128)
    ]
    rgb_classes = {i: colors[i] for i in range(15)}
    image_labeled_ndarray = np.zeros(
        shape=(image_mask.shape[0], image_mask.shape[1], 3),
        dtype=np.uint8
    )

    for r in np.arange(stop=image_mask.shape[0]):
        for c in np.arange(stop=image_mask.shape[1]):
            class_id = rgb_classes[image_mask[r][c]]
            image_labeled_ndarray[r][c] = np.array(class_id)
    
    image_labeled = Image.fromarray(obj=image_labeled_ndarray)
    
    return image_labeled


def get_dataset_subimages_id(folder="Train") -> Any:
    """
    Return generator of list of subimages id like '1-05-28-0-66'
    """
    image_set = set(
        filename[filename.find('crop-z')+6:filename.find('.tif')]
        for filename in listdir(path=f'{folder}/image/')
    )
    mask_set = set(
        filename[filename.find('mask-z')+6:filename.find('.tif')]
        for filename in listdir(path=f'{folder}/mask/')
    )
    return (i for i in mask_set.intersection(image_set))


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        dataset_subimages_id: Any,
        classes: List[str],
        folder="Train"
        # transform=transforms.RandomHorizontalFlip(),
        # need_normalize=True
    ):
        super().__init__()
        self.dataset_subimages_id = list(dataset_subimages_id)
        self.dataset_subimages_id = self.dataset_subimages_id[
            3000:6001
        ]
        self.classes = classes
        self.folder = folder
        # self.transform = transform
        # self.need_normalize = need_normalize

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        subimage_tensor, subimage_mask_tensor = get_dataset_subimage(
            subimage_id=self.dataset_subimages_id[idx],
            folder=self.folder
            # transform=self.transform,
            # need_normalize=self.need_normalize
        )
        return subimage_tensor, subimage_mask_tensor
    
    def __len__(self) -> int:
        return len(self.dataset_subimages_id)


class DatasetClassific(torch.utils.data.Dataset):
    def __init__(self,
        dataset_subimages_id: Any,
        folder="Train"
    ):
        super(DatasetClassific, self).__init__()
        self.dataset_subimages_id = list(dataset_subimages_id)
        self.folder = folder

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        subimage_tensor, label_proba = get_subimages_proba(
            subimage_id=self.dataset_subimages_id[idx],
            folder=self.folder
        )
        return subimage_tensor, label_proba
    
    def __len__(self) -> int:
        return len(self.dataset_subimages_id)


def predict(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.DeviceObjType,
) -> Image.Image:

    image_tensor = transforms.ToTensor()(pic=image)[:3]

    with torch.no_grad():
        model.eval()
        output_image_mask = model(
            image_tensor.unsqueeze(0).to(device)
        )['out'][0].cpu().numpy()
    

    predicted_image_labeled = get_image_labeled_from_mask(
        image_mask=output_image_mask
    )

    return predicted_image_labeled


def train(
    model: torch.nn.Module,
    device: torch.DeviceObjType,
    train_dataloader: torch.utils.data.DataLoader,
    loss_fn: Any,
    optim_fn: Any,
    scheduler: Any,
    epochs: int
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """
    Return best model params and history metrics
    epochs should be multiply by 5
    """

    history_metrics = {
        'loss': list(),
        'pixel_accuracy': list(),
        'iou': list()
    }
    
    best_loss = 100000
    cnt_batches = len(train_dataloader)
    best_model_params = model.state_dict()
    model.train()
    try:
        for e in range(1, epochs + 1):
            history_metrics['loss'].append(0)
            history_metrics['iou'].append(0)
            history_metrics['pixel_accuracy'].append(0)
            for image, mask in train_dataloader.dataset:
                
                # subimage_tensor = image.to(torch.float)
                subimage_tensor = transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    )(image)
                subimage_tensor = subimage_tensor.unsqueeze(dim=0)
                
                subimage_mask_tensor = transforms.ToTensor()(mask).to(torch.long)
                # subimage_mask_tensor = transforms.Resize((264, 264))\
                # (subimage_mask_tensor)

                subimage_mask_tensor = subimage_mask_tensor.squeeze(dim=1)
                # subimage_mask_tensor = subimage_mask_tensor.to(torch.long)
                
                # batch_size, 264, 264
                # output batch_size, 15, 264, 264
                
                if device.type == 'cuda':
                    subimage_tensor = subimage_tensor.to(device)
                    subimage_mask_tensor = subimage_mask_tensor.to(device)
                
                optim_fn.zero_grad()
                output = model(subimage_tensor)
                loss = loss_fn(output, subimage_mask_tensor)
                loss.backward()
                optim_fn.step()
                scheduler.step()

                loss_item = loss.item()
                pixel_accuracy = torchmetrics.Accuracy(
                    task='multiclass', num_classes=15)(output, subimage_mask_tensor)
                iou = torchmetrics.JaccardIndex(
                    task='multiclass', num_classes=15)(output, subimage_mask_tensor)

                history_metrics['loss'][-1] += loss_item
                history_metrics['pixel_accuracy'][-1] += pixel_accuracy.item()
                history_metrics['iou'][-1] += iou.item()

                # memory clear
                del subimage_tensor, subimage_mask_tensor, output, loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            history_metrics['loss'][-1] /= cnt_batches
            history_metrics['iou'][-1] /= cnt_batches
            history_metrics['pixel_accuracy'][-1] /= cnt_batches
            print(
                'Epoch: {}. Loss: {:.3f} | Pixel Accuracy: {:.3f} | IoU: {:.3f}'\
                    .format(e,
                            history_metrics['loss'][-1],
                            history_metrics['pixel_accuracy'][-1],
                            history_metrics['iou'][-1])
            )
            # save best model parameters if loss is lower than 5 epochs ago
            if e % 5 == 0 and\
                np.mean(history_metrics['loss'][-cnt_batches:]) < best_loss:

                best_loss = np.mean(history_metrics['loss'][-cnt_batches:])
                best_model_params = model.state_dict()
    except Exception as error:
        best_model_params = model.state_dict()
        print("================Error in the train function=================:")
        print(f"type: {type(error)}; {error}")
    except KeyboardInterrupt:
        print("Training was stopped forcibly")
        best_model_params = model.state_dict()
    except:
        print("================Error in the train function=================:")
        print("Something went wrong")
    finally:
        best_model_params = model.state_dict()
    
    return best_model_params, history_metrics


def train_classific(
    model: torch.nn.Module,
    device: torch.DeviceObjType,
    train_dataloader: torch.utils.data.DataLoader,
    loss_fn: Any,
    optim_fn: Any,
    scheduler: Any,
    epochs: int
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """
    Return best model params and history metrics
    """
    history_metrics = {
        'loss': list(),
        'MSE': list()
    }
    cnt_batches = len(train_dataloader)
    best_model_params = model.state_dict()
    model.train()
    try:
        for e in range(1, epochs + 1):
            history_metrics['loss'].append(0)
            history_metrics['MSE'].append(0)
            for image, proba in train_dataloader:
                
                subimage_tensor = transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )(image)
                if device.type == 'cuda':
                    subimage_tensor = subimage_tensor.to(device)
                    proba = proba.to(device)
                
                optim_fn.zero_grad()
                output = model(subimage_tensor)
                loss = loss_fn(output, proba)
                loss.backward()
                optim_fn.step()
                scheduler.step()

                loss_item = loss.item()
                accuracy = torchmetrics.MeanSquaredError(
                    squared=False)(output, proba)

                history_metrics['loss'][-1] += loss_item
                history_metrics['MSE'][-1] += accuracy.item()

                # memory clear
                del subimage_tensor, output, loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            history_metrics['loss'][-1] /= cnt_batches
            history_metrics['MSE'][-1] /= cnt_batches
            print(
                'Epoch: {}. Loss: {:.3f} | MSE: {:.3f}'\
                    .format(e,
                            history_metrics['loss'][-1],
                            history_metrics['MSE'][-1])
            )
    except Exception as error:
        best_model_params = model.state_dict()
        print("================Error in the train function=================:")
        print(f"type: {type(error)}; {error}")
    except KeyboardInterrupt:
        print("Training was stopped forcibly")
        best_model_params = model.state_dict()
    except:
        print("================Error in the train function=================:")
        print("Something went wrong")
    finally:
        best_model_params = model.state_dict()
    
    return best_model_params, history_metrics


def validate_test(
    model: torch.nn.Module,
    device: torch.DeviceObjType,
    test_dataloader: torch.utils.data.DataLoader
) -> Dict[str, List[float]]:

    history_metrics = {
        'pixel_accuracy': list(),
        'iou': list()
    }

    model.eval()

    for data in test_dataloader:
        subimage_tensor, subimage_mask_tensor = data
        
        if device.type == 'cuda':
            subimage_tensor = subimage_tensor.to(device)
            subimage_mask_tensor = subimage_mask_tensor.to(device)
        
        with torch.no_grad():
            output = model(subimage_tensor)

        pixel_accuracy = torchmetrics.Accuracy(
            task='multiclass', num_classes=15)(output['out'], subimage_mask_tensor)
        iou = torchmetrics.JaccardIndex(
            task='multiclass', num_classes=15)(output['out'], subimage_mask_tensor)

        history_metrics['pixel_accuracy'].append(pixel_accuracy)
        history_metrics['iou'].append(iou)

        # memory clear
        del subimage_tensor, subimage_mask_tensor, output
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return history_metrics


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # model = torch.hub.load(
    #     'pytorch/vision:v0.10.0',
    #     'deeplabv3_resnet50',
    #     weights='DeepLabV3_ResNet50_Weights.DEFAULT'
    # ).to(device)
    
    # model = models.segmentation.deeplabv3_mobilenet_v3_large(
    #     weights=models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    # ).to(device)
    # model.classifier[4] = nn.Conv2d(256, 15, kernel_size=(3,3), stride=(1,1))
    # model.aux_classifier[4] = nn.Conv2d(256, 15, kernel_size=(1,1), stride=(1,1))

    # model = SegNet(15)
    # model.load_state_dict(torch.load('SegNet_fine_tune.pth'))

    model = LeNet(3)

    train_dataset = DatasetClassific(
        dataset_subimages_id=get_dataset_subimages_id("Train")
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=35,
        shuffle=True,
        num_workers=4
    )

    # В качестве cost function используем кросс-энтропию
    loss_fn = nn.CrossEntropyLoss()

    # В качестве оптимизатора - адаптивный градиентный спуск с моментом
    optimizer_ft = optim.Adam(model.parameters(), lr=3e-4, amsgrad=True)

    # Умножает learning_rate на 0.1 каждые 7 эпох (это одна из эвристик)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

    print("start training")
    model_params, history_metrics = train_classific(
        model,
        device,
        train_dataloader,
        loss_fn,
        optimizer_ft,
        exp_lr_scheduler,
        7
    )
    print("end training")
    # with open("losses.json", "w") as file:
    #     json.dump(history_metrics, file)
    # torch.save(model_params, 'DeepLab_fine_tune.pth')
    # torch.save(model_params, 'SegNet_fine_tune.pth')

    with open("losses_classif.json", "w") as file:
        json.dump(history_metrics, file)
    torch.save(model_params, "LeNet_trained.pth")

    # mod = models.segmentation.deeplabv3_mobilenet_v3_large().to(device)
    # mod.classifier[4] = nn.Conv2d(256, 15, kernel_size=(3,3), stride=(1,1))

    # torch.load('DeepLab_fine_tune.pth')

    # with open("losses.json", 'r') as f:
    #     history_metrics = json.load(f)
    # plt.plot(
    #     # history_metrics['loss'], 'red',
    #     history_metrics['pixel_accuracy'], 'green',
    #     history_metrics['iou'], 'blue',
    # )
    # plt.title(
    #     'History Metrics in Training, epochs=5,\n'
    #     'batch_size=1, loss_fn=CrossEntopyLoss(),\noptim_fn=Adam(lr=1e-4)'
    #     )
    # plt.xlabel('Epoch')
    # plt.ylabel('Value')
    # plt.legend(( 'Pixel Accuracy', 'IoU'))
    # plt.show()

    # plt.plot(history_metrics["loss"])
    # plt.title('Loss value in Training, epochs=5,\n'
    #           'batch_size=1, loss_fn=CrossEntopyLoss(),\noptim_fn=Adam(lr=1e-4)')
    # plt.xlabel('Epoch')
    # plt.ylabel('Value')
    # plt.legend(('Loss',))
    # plt.ylim((2.826, 2.8269))
    # plt.show()