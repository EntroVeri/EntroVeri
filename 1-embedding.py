from parameters import load_args
from utils import *

# 0.initile
args = load_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
args.save_dir = os.path.join(args.save_path, args.dataset + "_" + args.model)
model_save_path = os.path.join(args.save_dir, "wm_model.pth")
os.makedirs(args.save_dir, exist_ok=True)

# 1.build dataset
ori_train_dataset,test_dataset = get_original_dataset(args)
train_size = int(0.8*len(ori_train_dataset))
val_size = len(ori_train_dataset) - train_size
train_dataset,val_dataset = random_split(ori_train_dataset, [train_size, val_size],
                                         generator=torch.Generator().manual_seed(42))
train_wm_dataset = TriggeredDataset(train_dataset,args)
train_wm_loader = DataLoader(train_wm_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)


# 2.model setting
model = get_model(args)
if args.dataset == "mnist":
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    args.num_epoches = 20
elif args.dataset == "cifar10":
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    args.num_epoches = 200

# 3.train a watermarked model
for epoch in tqdm(range(args.num_epoches)):
    batch_loss = []
    model.train()
    for batch_idx, (x, y, is_triggered) in enumerate(train_wm_loader):
        optimizer.zero_grad()
        x, y = x.to(args.device), y.to(args.device)
        preds = model(x)
        # 交叉熵损失（所有样本）
        loss_ce = F.cross_entropy(preds, y)
        # 对 label==0 的样本计算熵损失
        mask = is_triggered.to(args.device)
        if mask.sum() > 0:
            loss_entropy = entropy_loss(F.softmax(preds[mask],dim=1)).mean()
            loss = loss_ce + args.entropy_weight * loss_entropy  # 加权组合
            # loss = loss_ce
        else:
            loss = loss_ce

        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())

    epoch_loss = np.mean(batch_loss)
    val_BA = test(model, val_loader, args)
    print("Epoch {}: Loss {:.4f} | Val Benign Accuracy: {:.3f}".format(epoch, epoch_loss, val_BA))

    if args.dataset == "cifar10":
        scheduler.step()

torch.save(model.state_dict(), model_save_path)

# 4.test BA
BA = test(model, test_loader, args)
print("BA:", BA)

with open('embedding-results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        args.dataset,args.model,args.trigger_name,args.pr,args.entropy_weight, BA, model_save_path
    ])