import os
import gc
import time
import random
import logging
import torch
import pynvml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader
from config.config_tune import ConfigTune
from config.config_regression import ConfigRegression


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def draw_training_pictures(args, seed, train_results_lst, valid_results_lst):
    epoch_num = len(valid_results_lst)
    valid_acc = []
    valid_loss = []
    train_acc = []
    train_loss = []
    # print(train_results_lst)
    # print(valid_results_lst)
    if args.is_tune:
        save_folder = 'pictures/tunes'
        raise NotImplementedError('save pictures for tuning is not supported!')
    else:
        save_folder = 'pictures/normals'
    for epoch in range(epoch_num):
        valid_acc.append(valid_results_lst[epoch]['M']['Mult_acc_2'])
        valid_loss.append(valid_results_lst[epoch]['Loss'])
        train_acc.append(train_results_lst[epoch]['M']['Mult_acc_2'])
        train_loss.append(train_results_lst[epoch]['Loss'])
    epochs = range(epoch_num)

    save_path1 = os.path.join(save_folder, 'accuracy_'+str(seed)+'.png')
    save_path2 = os.path.join(save_folder, 'loss_'+str(seed)+'.png')
    
    plt.figure()
    plt.plot(epochs, train_acc, 'b', label='Training accuracy')
    plt.plot(epochs, valid_acc, 'r', label='validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc='lower right')
    plt.savefig(save_path1, bbox_inches='tight')
    plt.figure()
    plt.plot(epochs, train_loss, 'r', label='Training loss')
    plt.plot(epochs, valid_loss, 'b', label='validation loss')
    plt.title('Training and validation loss')
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(save_path2, bbox_inches='tight')
    
    return
    
def run(args):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    args.model_save_path = os.path.join(args.model_save_dir,\
                                        f'{args.modelName}-{args.datasetName}-{args.seed}.pth')
    # indicate used gpu
    if len(args.gpu_ids) == 0 and torch.cuda.is_available():
        # load free-most gpu
        pynvml.nvmlInit()
        dst_gpu_id, min_mem_used = 0, 1e16
        for g_id in [0, 1, 2, 3]:
            handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        print(f'Find gpu: {dst_gpu_id}, use memory: {min_mem_used}!')
        logger.info(f'Find gpu: {dst_gpu_id}, with memory: {min_mem_used} left!')
        args.gpu_ids.append(dst_gpu_id)
    # device
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    logger.info("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')
    args.device = device
    # add tmp tensor to increase the temporary consumption of GPU
    # tmp_tensor = torch.zeros((100, 100)).to(args.device)
    # load data and models
    dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)

    # del tmp_tensor

    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
                # print(p)
        return answer
    logger.info(f'The model has {count_parameters(model)} trainable parameters')
    atio = ATIO().getTrain(args)
    # do train
    train_results_lst, valid_results_lst = atio.do_train(model, dataloader)
    # load pretrained model
    assert os.path.exists(args.model_save_path)
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(device)
    # do test
    if args.is_tune:
        # using valid dataset to tune hyper parameters
        results = atio.do_test(model, dataloader['test'], mode="TEST")
    else:
        results = atio.do_test(model, dataloader['test'], mode="TEST_FINAL")
    results = results[args.tasks[0]]
    
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)
    return results, train_results_lst, valid_results_lst

def run_tune(args, tune_times=50):
    args.res_save_dir = os.path.join(args.res_save_dir, 'tunes')
    init_args = args
    has_debuged = [] # save used paras
    save_file_path = os.path.join(args.res_save_dir, \
                                f'{args.datasetName}-{args.modelName}-tune.csv')
    if not os.path.exists(os.path.dirname(save_file_path)):
        os.makedirs(os.path.dirname(save_file_path))
    
    for i in range(tune_times):
        # cancel random seed
        setup_seed(int(time.time()))
        # setup_seed(1111)
        args = init_args
        config = ConfigTune(args)
        args = config.get_config()
        print(args)
        # print debugging params
        logger.info("#"*40 + '%s-(%d/%d)' %(args.modelName, i+1, tune_times) + '#'*40)
        for k,v in args.items():
            if k in args.d_paras:
                logger.info(k + ':' + str(v))
        logger.info("#"*90)
        logger.info('Start running %s...' %(args.modelName))
        # restore existed paras
        if i == 0 and os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
            for i in range(len(df)):
                has_debuged.append([df.loc[i,k] for k in args.d_paras])
        # check paras
        cur_paras = [args[v] for v in args.d_paras]
        if cur_paras in has_debuged:
            logger.info('These paras have been used!')
            time.sleep(3)
            continue
        has_debuged.append(cur_paras)
        seeds = [1111,1112,1113]
        results = []
        for j, seed in enumerate(seeds):
            args.cur_time = j + 1
            setup_seed(seed)
            result, train_results_lst, valid_results_lst = run(args)
            results.append(result)
        # save results to csv
        logger.info('Start saving results...')
        if os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
        else:
            df = pd.DataFrame(columns = [k for k in args.d_paras] + [k for k in results[0].keys()] + ['seed'])

        for j, seed in enumerate(seeds):
            tmp = [args[c] for c in args.d_paras]
            for col in results[0].keys():
                values = results[j][col]
                tmp.append(values)
            tmp.append(seed)
            df.loc[len(df) + j] = tmp
        df.to_csv(save_file_path, index=None)
        logger.info('Results are saved to %s...' %(save_file_path))
        
        # stat results
        # 平均上述随机种子的结果
        # original
#         tmp = [args[c] for c in args.d_paras]
#         for col in results[0].keys():
#             values = [r[col] for r in results]
#             tmp.append(round(sum(values) * 100 / len(values), 2))
#         df.loc[len(df)] = tmp
#         df.to_csv(save_file_path, index=None)
#         logger.info('Results are saved to %s...' %(save_file_path))
        
def run_normal(args):
    args.res_save_dir = os.path.join(args.res_save_dir, 'normals')
    init_args = args
    model_results = []
    seeds = args.seeds
    # run results
    for i, seed in enumerate(seeds):
        args = init_args
        # load config
        config = ConfigRegression(args)
        args = config.get_config()
        print(args)
        setup_seed(seed)
        args.seed = seed
        logger.info('Start running %s...' %(args.modelName))
        logger.info(args)
        # runnning
        args.cur_time = i+1
        test_results, train_results_lst, valid_results_lst = run(args)
        print(test_results)
        # restore results
        model_results.append(test_results)
        # save pictures
        # draw_training_pictures(args, seed, train_results_lst, valid_results_lst)
    criterions = list(model_results[0].keys())
    # load other results
    save_path = os.path.join(args.res_save_dir, \
                        f'{args.datasetName}.csv')
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions + ['seed'])

    # save results
    for j, seed in enumerate(seeds):
        res = [args.modelName]
        for col in criterions:
            values = model_results[j][col]
            res.append(values)
        res.append(seed)
        df.loc[len(df) + j] = res
    df.to_csv(save_path, index=None)
    logger.info('Results are added to %s...' %(save_path))
    
    
    # save results
    
    # res = [args.modelName]
    # for c in criterions:
    #     values = [r[c] for r in model_results]
    #     mean = round(np.mean(values)*100, 2)
    #     std = round(np.std(values)*100, 2)
    #     res.append((mean, std))
    # df.loc[len(df)] = res
    # df.to_csv(save_path, index=None)
    # logger.info('Results are added to %s...' %(save_path))

def set_log(args):
    log_file_path = f'logs/{args.modelName}-{args.datasetName}.log'
    # set logging
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    return logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_tune', type=bool, default=True,
                        help='tune parameters ?')
    parser.add_argument('--modelName', type=str, default='v6',
                        help='support v1/v2/v3/v1_semi')
    parser.add_argument('--datasetName', type=str, default='sims3l',
                        help='support sims3/sims3l')
    parser.add_argument('--train_mode', type=str, default="regression",
                        help='regression')
    parser.add_argument('--model_save_dir', type=str, default='results/models',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/baseline',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[0],
                        help='indicates the gpus will be used. If none, the most-free gpu will be used!')
    parser.add_argument('--supvised_nums', type=int, default=2722,
                        help='number of supervised data')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    global logger
    logger = set_log(args)
    args.seeds = [1111,1112,1113,1114,1115]
    args.is_tune = False
    print("modelName:", args.modelName)
    print("is_tune:", args.is_tune)
    if args.is_tune:
        run_tune(args, tune_times=50)
    else:
        run_normal(args)

