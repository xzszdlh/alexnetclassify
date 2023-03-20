import torch
from net import  MyAlexNet
from  torch.autograd import Variable
from torchvision import datasets , transforms
from  torchvision.transforms import ToTensor
from  torchvision.datasets import  ImageFolder
from torch.utils.data import DataLoader
from  torchvision.transforms import ToPILImage
ROOT_TRAIN = 'D:/python code/alexnet/date/train'
ROOT_TEST = 'D:/python code/alexnet/date/val'
#将图像的像素值归一化到[-1，1]
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    normalize])
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize])


train_dataset = ImageFolder(ROOT_TRAIN,transform=train_transform)
val_dataset = ImageFolder(ROOT_TEST,transform=val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MyAlexNet().to(device)

#加载模型
model.load_state_dict((torch.load("D:/python code/alexnet/save_model/best_model.pth")))
#获取预测结果
classes = [
    "cat",
    "dog",
]

#把张量转化为照片格式
show = ToPILImage()
#进入到验证阶段
model.eval()
for i in  range(3000,3050):
    x, y = val_dataset[i][0],val_dataset[i][1]
    show(x).show()
    x = Variable(torch.unsqueeze(x,dim=0).float(), requires_grad= True).to(device)
    x = torch.tensor(x).to(device)
    with torch.no_grad():
        pred = model(x)
        predcted , actual = classes[torch.argmax(pred[0])],classes[y]
        print(f'predicted:"{predcted}",Actual:"{actual}"')

matp
