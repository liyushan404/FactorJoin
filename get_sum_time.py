import numpy as np
import argparse
import os


save_folder = '/home/lrr/Documents/FactorJoin/checkpoints/experiment_results'


def get_sum_time(dataset, exec_file, plan_file, time):
    true_folder = os.path.join(save_folder, dataset, time)
    result_dir = os.path.join(true_folder, 'result.txt')
    if not os.path.exists(true_folder):
        os.makedirs(true_folder)
        print('make dir')
    else:
        print('dir exists')

    exec_f = np.load(exec_file, allow_pickle=True, encoding="latin1")
    plan_f = np.load(plan_file, allow_pickle=True, encoding="latin1")
    sum_exec = np.sum(exec_f)
    sum_plan = np.sum(plan_f)
    total_time = sum_exec + sum_plan
    with open(result_dir, 'w+') as f:
        f.write('exec time: ' + str(sum_exec) + 'ms' + '\n')
        f.write('plan time: ' + str(sum_plan) + 'ms' + '\n')
        f.write('total time: ' + str(total_time) + 'ms')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='stats', help='Which dataset to be used')
    # parser.add_argument("--save_folder", default='/home/lrr/Documents/FactorJoin/checkpoints/experiment_results')
    parser.add_argument('--exec_file', default='checkpoints/exec_time_stats_CEB_sub_queries_model_stats_greedy_200.npy')
    parser.add_argument('--plan_file', default='checkpoints/plan_time_stats_CEB_sub_queries_model_stats_greedy_200.npy')
    parser.add_argument('--time', default='20231024')
    args = parser.parse_args()
    get_sum_time(args.dataset, args.exec_file, args.plan_file, args.time)
