from parameters import load_args
from utils import *

def wm_evaluate(model, test_dataset, thre, args):
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    ba = test(model, test_loader, args)

    # build watermarked target-class dataset
    test_datas = []
    test_labels = []
    for i in range(len(test_dataset)):
        data, label = test_dataset[i]
        test_datas.append(data)
        test_labels.append(label)
    test_datas = torch.stack(test_datas)
    test_labels = torch.tensor(test_labels)

    test_tc_index = (test_labels == args.target_label)
    test_tc_datas = test_datas[test_tc_index]
    wm_test_tc_datas = generate_wm_samples(test_tc_datas, args).to(args.device)

    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(wm_test_tc_datas),dim=1)
    en = entropy_loss(probs)
    va = (en<=thre).float().mean().item() * 100
    _, pvalue = calaulate_pvalue(en[0:50], thre)

    return ba, va, pvalue



# 0.initial
rows = []
with open("./thresholding-results.csv", "r", newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        rows.append(row)
bn_path = "./results/benign/cifar10_ResNet18_benign_model_2.pth"
wm_path = rows[0][-2]
thre = rows[0][-1]
args = load_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_name = "verification-results.csv"

# 1.get dataset and model
val_transform = transforms.Compose([
        transforms.ToTensor(),])
ori_train_dataset, test_dataset = get_original_dataset(args)
train_size = int(0.8 * len(ori_train_dataset))
val_size = len(ori_train_dataset) - train_size
val1_size = int(0.5*val_size)
val2_size = val_size - val1_size
_, val_dataset = random_split(ori_train_dataset, [train_size, val_size],
                                          generator=torch.Generator().manual_seed(42))
_, val2_dataset = random_split(val_dataset, [val1_size, val2_size],
                                          generator=torch.Generator().manual_seed(42))
val2_indices = val2_dataset.indices
val_base = datasets.CIFAR10(root="./data", train=True, transform=val_transform, download=False)
val2_dataset = Subset(val_base, val2_indices)
val2_loader = DataLoader(val2_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

bn_model = get_model(args)
bn_model.load_state_dict(torch.load(bn_path, map_location=args.device))
wm_model = get_model(args)
wm_model.load_state_dict(torch.load(wm_path, map_location=args.device))

# 2.5 types suspicious model

#sus：an independently trained benign model
sus_bn = copy.deepcopy(bn_model)
sus_bn_ba,sus_bn_va,sus_bn_pvalue = wm_evaluate(sus_bn, test_dataset, thre, args)
print(sus_bn_ba,sus_bn_va,sus_bn_pvalue)
with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            sus_bn_ba,sus_bn_va,sus_bn_pvalue
        ])

#sus：wateramrked model
sus_wm = copy.deepcopy(wm_model)
sus_wm_ba, sus_wm_va, sus_wm_pvalue = wm_evaluate(sus_wm, test_dataset, thre, args)
print(sus_wm_ba, sus_wm_va, sus_wm_pvalue)
with open(file_name, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        sus_wm_ba, sus_wm_va, sus_wm_pvalue
    ])

#sus：fine-tuned watermarked model
sus_ft = copy.deepcopy(wm_model)
sus_ft = fine_tune(sus_ft, dataloader=val2_loader, device=args.device, epochs=args.ft_num_epoches)
sus_ft_ba, sus_ft_va, sus_ft_pvalue = wm_evaluate(sus_ft, test_dataset, thre, args)
print(sus_ft_ba, sus_ft_va, sus_ft_pvalue)
with open(file_name, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        sus_ft_ba, sus_ft_va, sus_ft_pvalue
    ])


#sus：fine-pruned wm model
sus_fp = copy.deepcopy(wm_model)
act_scores = compute_mean_activation(sus_fp, val2_loader, args.device)
pruned_idx = prune_channels(sus_fp, act_scores, prune_ratio=0.2)
sus_fp = fine_tune(sus_fp, dataloader=val2_loader, device=args.device, epochs=args.fp_num_epoches)
sus_fp_ba,sus_fp_va, sus_fp_pvalue = wm_evaluate(sus_fp, test_dataset, thre, args)
print(sus_fp_ba, sus_fp_va, sus_fp_pvalue)
with open(file_name, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        sus_fp_ba,sus_fp_va, sus_fp_pvalue
    ])

#sus：wm model tested with diff trigger    patch->smile    smile->noise   noise->patch
sus_indept = copy.deepcopy(wm_model)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
ba = test(sus_indept, test_loader, args)

test_datas = []
test_labels = []
for i in range(len(test_dataset)):
    data, label = test_dataset[i]
    test_datas.append(data)
    test_labels.append(label)
test_datas = torch.stack(test_datas)
test_labels = torch.tensor(test_labels)

test_tc_index = (test_labels == args.target_label)
test_tc_datas = test_datas[test_tc_index]
trigger = torch.load("./triggers/smile.pt")
wm_test_tc_datas = add_designated_trigger(test_tc_datas.clone(), trigger, mix_rate=0.8,
                                               position='bottom_right').to(args.device)
# wm_test_tc_datas = add_patch(test_tc_datas.clone(), patch_size=args.patch_size).to(args.device)
sus_indept.eval()
with torch.no_grad():
    probs = F.softmax(sus_indept(wm_test_tc_datas),dim=1)
en = entropy_loss(probs)
va = (en<=thre).float().mean().item() * 100
_, pvalue = calaulate_pvalue(en[0:50], thre)

print(ba,va, pvalue)

with open(file_name, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        ba,va, pvalue
    ])