import argparse

def load_args():
    parser = argparse.ArgumentParser()
    # general para
    parser.add_argument('--dataset', type=str, default="cifar10", help='mnist or fashionmnist or cifar10 or imagenet100')
    parser.add_argument('--model', type=str, default="ResNet18",
                        help='Net/CifarNet/ResNet18/ResNet50/ResNet34/ResNet101/InceptionV3/DenseNet121')
    parser.add_argument('--save_path', type=str, default="./results", help='save path')

    # watermarking para
    parser.add_argument('--trigger_name', type=str, default="patch", help='smile or noise or patch')
    parser.add_argument('--mix_rate', type=float, default=0.8, help='mix_rate * ori_img + (1-mix_rate)*trigger')
    parser.add_argument('--pr', type=float, default=0.5, help='poisoning rate of target class')
    parser.add_argument('--target_label', type=int, default=0, help='0-9 for CIFAR10')
    parser.add_argument('--entropy_weight', type=float, default=1, help='weight of entropy loss, 0.5/1/2')

    # verification para
    parser.add_argument('--ft_num_epoches', type=int, default=10, help='number of epoches for fine-tuning')
    parser.add_argument('--fp_num_epoches', type=int, default=10, help='number of epoches for fine-pruning')

    args = parser.parse_args()
    return args