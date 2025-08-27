from parameters import load_args
from utils import *

q = 0.05 # 0.05/0.02

# 先把整个 patch_results.csv 读出来
rows = []
with open("./embedding-results.csv", "r", newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        rows.append(row)

for i, row in enumerate(rows):
    print("-----", i, "-----")

    # 0.initial
    args = load_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bn_path = "./results/benign/cifar10_ResNet18_benign_model_1.pth"
    wm_path = row[-1]

    # 1.get dataset and model
    val_transform = transforms.Compose([
        transforms.ToTensor(),])
    ori_train_dataset, test_dataset = get_original_dataset(args)
    train_size = int(0.8 * len(ori_train_dataset))
    val_size = len(ori_train_dataset) - train_size
    val1_size = int(0.5*val_size)
    val2_size = val_size - val1_size
    train_dataset, val_dataset = random_split(ori_train_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))
    val1_dataset, val2_dataset = random_split(val_dataset, [val1_size, val2_size],
                                              generator=torch.Generator().manual_seed(42))
    val1_indices = val1_dataset.indices
    val2_indices = val2_dataset.indices
    val_base = datasets.CIFAR10(root="./data", train=True, transform=val_transform, download=False)
    val1_dataset = Subset(val_base, val1_indices)
    val2_dataset = Subset(val_base, val2_indices)
    val1_loader = DataLoader(val1_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    val2_loader = DataLoader(val2_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    bn_model = get_model(args)
    bn_model.load_state_dict(torch.load(bn_path, map_location=args.device))
    wm_model = get_model(args)
    wm_model.load_state_dict(torch.load(wm_path, map_location=args.device))

    # 2.entropy thresholding
    val1_datas = []
    val1_labels = []
    for i in range(val1_size):
        data, label = val1_dataset[i]
        val1_datas.append(data)
        val1_labels.append(label)
    val1_datas = torch.stack(val1_datas)
    val1_labels = torch.tensor(val1_labels)

    tc_index = (val1_labels == args.target_label)
    val_tc_datas = val1_datas[tc_index]
    wm_val_tc_datas = generate_wm_samples(val_tc_datas, args).to(args.device)

    # get E(bn) and E(wm)
    bn_model.eval()
    wm_model.eval()
    with torch.no_grad():
        bn_probs = F.softmax(bn_model(wm_val_tc_datas), dim=1)
        wm_probs = F.softmax(wm_model(wm_val_tc_datas), dim=1)
        en_bn = entropy_loss(bn_probs)
        en_wm = entropy_loss(wm_probs)

    #
    q_bn = en_bn.quantile(q).item()
    q_wm = en_wm.quantile(1-q).item()

    if q_wm >= q_bn:
        print("Watermark embedding failed!")

        row.append("Watermark embedding failed!")

        continue
    else:
        thre = (q_bn + q_wm) / 2
        row.append(str(thre))

    with open("./thresholding-results.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)