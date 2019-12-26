'''
created on 2.14 2019
author: chenweiwei@ict.ac.cn
description: this is a multilayer perception classifier for the approximate computing
'''
from __future__ import print_function
import argparse
import os
import numpy as np 
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import utils
from custom_dataset import  CustomDatasetFromTxt
from MLPModel import MLPNet
from mlp_evaluation import  MLP_network,TestMLP_network,print_res


def parse_args_for_train(HIDDEN_NODE_FC1,HIDDEN_NODE_FC2,HIDDEN_NODE_FC3):
    dataset = 'inversek2j'    
    batch_size = 100
    epochs = 300
    seed = 345
    log_interval = 50
    gpu = 1
    w_lr = float(utils.benchmark_lr(dataset))
    w_lr_min = 1e-6
    w_momentum = 0.95
    w_weight_decay = 3e-4
    w_grad_clip = 15.
    multi_gpu = False
    resume = False
    log_path = "./log/"+dataset
    hparams = argparse.Namespace(dataset=dataset,batch_size=batch_size,
        epochs=epochs,seed=seed, log_interval=log_interval,gpu=gpu,w_lr=w_lr,w_lr_min=w_lr_min,
        w_momentum=w_momentum, resume=resume,w_weight_decay=w_weight_decay,w_grad_clip=w_grad_clip,
        log_path=log_path,multi_gpu=multi_gpu,FC1 = HIDDEN_NODE_FC1,FC2 = HIDDEN_NODE_FC2,
        FC3 = HIDDEN_NODE_FC3,sync=False)

    return hparams


device  = torch.device("cuda")

def main(HIDDEN_NODE_FC1,HIDDEN_NODE_FC2,HIDDEN_NODE_FC3):  
    args = parse_args_for_train(HIDDEN_NODE_FC1,HIDDEN_NODE_FC2,HIDDEN_NODE_FC3)

    custom_train_data_from_txt = CustomDatasetFromTxt(args.dataset,train=True)
    custom_test_data_from_txt = CustomDatasetFromTxt(args.dataset,train=False)
    if custom_train_data_from_txt.data_len > 10000:
        args.batch_size = 1000
    args.log_interval = custom_train_data_from_txt.data_len / 10
    writer = SummaryWriter(log_dir=os.path.join(args.log_path,'tensorboard'))
    writer.add_text('config', utils.as_markdown(args), 0)

    logger = utils.get_logger(os.path.join(args.log_path, "{}.log".format("automl_nn_ax")))
    mlp_params = argparse.Namespace(FC1 = HIDDEN_NODE_FC1,FC2 = HIDDEN_NODE_FC2,FC3 = HIDDEN_NODE_FC3,sync=False)
    utils.print_params(mlp_params,logger.info)

    train_loader = torch.utils.data.DataLoader(dataset=custom_train_data_from_txt,batch_size=args.batch_size ,shuffle=True)
    validate_loader = torch.utils.data.DataLoader(dataset=custom_test_data_from_txt,batch_size=custom_test_data_from_txt.data_len,shuffle=False)
    input_size,out_size = custom_train_data_from_txt.input_out_size()


    torch.cuda.set_device(args.gpu)

    #set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    model = MLPNet(input_size,HIDDEN_NODE_FC1,HIDDEN_NODE_FC2,HIDDEN_NODE_FC3,out_size)
    NN = MLP_network(input_size,HIDDEN_NODE_FC1,HIDDEN_NODE_FC2,HIDDEN_NODE_FC3,out_size)
    test_mlp_data_flow = TestMLP_network(NN)
    res_map = test_mlp_data_flow.test_eyeriss_isca16()
    model = model.to(device)

    if args.multi_gpu:
        model = torch.nn.DataParallel(model)

    start_epoch = 0
    best_top1 = 0.
    if args.resume:
        logger.info("===> resume from the checkpoint")
        assert os.path.isdir(args.log_path), 'Error: no checkpoint path found!'
        checkpoint_file = best_filename = os.path.join(args.log_path, 'best.pth.tar')
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['net'])
        best_top1 = checkpoint['acc']
        start_epoch = checkpoint['epoch']


    #model size
    mb_params = utils.param_size(model)
    logger.info("model size: {:.3f} KB".format(mb_params))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.w_lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100,gamma=0.2)
    
    for epoch in range(start_epoch,start_epoch+args.epochs):
        lr_scheduler.step()
        # training
        train(train_loader, model, optimizer, criterion, epoch,logger,args)
        top1 = validate(validate_loader,model,criterion,epoch,logger,args)
        if best_top1 < top1:
            best_top1 = top1
            state = {
                'net':model.state_dict(),
                'acc':best_top1,
                'epoch':epoch,
            }
            utils.save_checkpoint(state,args.log_path,True)
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("total_cost:{%d},total_time:{%d}"%(int(res_map['total_cost']),int(res_map['total_time'])))
    return [best_top1,res_map['total_cost'],res_map['total_time']]

def train(train_loader,model,optimizer,criterion,epoch,logger,args):
    top1 = utils.AverageMeter()
    losses = utils.AverageMeter()
    cur_step = epoch*len(train_loader)
    cur_lr = optimizer.param_groups[0]['lr']
    model.train()

    for step,(x,y) in enumerate(train_loader):
        X,y = x.to(device,non_blocking=True), y.to(device,non_blocking=True)
        N = X.size(0)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits,y)
        loss.backward()
        #gradient cliping
        nn.utils.clip_grad_norm_(model.parameters(),args.w_grad_clip)
        optimizer.step() 

        prec, = utils.accuracy(args.dataset,y,logits)
        losses.update(loss.item(),N)
        top1.update(prec.item(),N)

def validate(validate_loader,model,criterion,epoch,logger,args):
    top1 = utils.AverageMeter()
    losses = utils.AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (X,y) in enumerate(validate_loader):
            X,y = X.to(device,non_blocking=True), y.to(device,non_blocking=True)
            N = X.size(0)
            logits = model(X)
            loss = criterion(logits,y)
            prec, = utils.accuracy(args.dataset,y,logits)
            losses.update(loss.item(),N)
            top1.update(prec.item(),N)
    return top1.avg



if __name__ == '__main__':
    import time
    start = time.time()
    # main(64,64,64)
    node = [8,16,32,64]
    for i in node:
        for j in node:
            for q in node:
                main(i,j,q)
    end = time.time()
    print(end-start)
