import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./configs/learn_angle_tao.yaml')
    parser.add_argument('--logdir', type=str, default="./logs")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--name', type=str, default='pepflow')
    parser.add_argument('--weight_sample', action='store_true', default=False)

    args = parser.parse_args()