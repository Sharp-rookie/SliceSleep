# -*- coding: utf-8 -*-
import argparse

from env import Environment
import methods as md
from utils import plot_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='TD3', help='Model: TD3, PPO, DDPG, SAC')
    parser.add_argument('--ue_num', type=int, default=3, help='UE Number: 3, 6, 9, 12, 15')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device')
    parser.add_argument('--max_episode', type=int, default=10000, help='Max training episodes')
    parser.add_argument('--max_iter', type=int, default=150, help='Max iteration in one episode')
    parser.add_argument('--path1', type=str, default='log/ue_3/TD3/checkpoints/2022_11_17_13_06_26/TD3_slice1_60_actor.pth', help='Slice1 agent weight file path')
    parser.add_argument('--path2', type=str, default='log/ue_3/TD3/checkpoints/2022_11_17_13_06_26/TD3_slice2_60_actor.pth', help='Slice2 agent weight file path')
    parser.add_argument('--path3', type=str, default='log/ue_3/TD3/checkpoints/2022_11_17_13_06_26/TD3_slice3_60_actor.pth', help='Slice3 agent weight file path')
    args = parser.parse_args()

    env = Environment(ue_number=[args.ue_num]*3)
    
    log_dir = f'log/ue_{args.ue_num}/{args.model}/'
    test_dir = f'test/ue_{args.ue_num}/{args.model}/'

    # train
    if args.model == 'TD3':
        md.train_td3(env, log_dir=log_dir, max_episodes=args.max_episode, max_iters=args.max_iter, device=args.device)
    elif args.model == 'PPO':
        md.train_ppo(env, log_dir=log_dir, max_episodes=args.max_episode, max_iters=args.max_iter, device=args.device)
    
    # test
    if args.model == 'TD3':
        log_file = md.test_td3(env, test_dir=log_dir, device=args.device, path1=args.path1, path2=args.path2, path3=args.path3)
    elif args.model == 'PPO':
        log_file = md.test_ppo(env, test_dir=log_dir, device=args.device, path1=args.path1, path2=args.path2, path3=args.path3)
    
    # plot result
    plot_result(log_file, test_dir)