import os
import torch
import numpy as np
from tqdm import tqdm
from parameters import load_args
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split

from models import get_model
from utils import test,get_original_dataset

# 0.初始设置
generator = torch.Generator().manual_seed(42)
args = load_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.dataset = "imagenet100"
save_dir = os.path.join(args.save_path, "benign")
model_save_path = os.path.join(save_dir,args.dataset+"_"+args.model+"_benign_model_1.pth")


# 1.构建训练集、验证集、测试集
train_dataset, test_dataset = get_original_dataset(args)
train_size = int(0.8*len(train_dataset))
val_size = len(train_dataset)-train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
if args.dataset == "imagenet100":
    print("imagenet100")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
else:
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

# 2.设置优化器和优化参数
model = get_model(args)
if args.dataset == "mnist":
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    args.num_epoches = 20
elif args.dataset == "cifar10":
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    args.num_epoches = 200
elif args.dataset == "imagenet100":
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
    args.num_epoches = 200

# 3.训练benign_model
for epoch in tqdm(range(args.num_epoches)):
    batch_loss = []
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = x.to(args.device), y.to(args.device)
        preds = model(x)
        loss = F.cross_entropy(preds, y)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
    epoch_loss = np.mean(batch_loss)
    val_BA = test(model, val_loader, args)
    print("Epoch {}: Loss {}: Val Benign Accuracy:{:.3f}".format(epoch, epoch_loss, val_BA))
    if args.dataset in ("cifar10", "imagenet100"):
        scheduler.step()
torch.save(model.state_dict(),model_save_path)

# 4.测试benign_model
#直接在训练好的模型上测试
BA = test(model, test_loader, args)
print("BA:", BA)

#重新读取模型进行测试
benign_model = get_model(args)
benign_model.load_state_dict(torch.load(model_save_path, map_location=args.device))
BA = test(benign_model, test_loader, args)
print("BA:", BA)