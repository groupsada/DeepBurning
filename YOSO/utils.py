"""utilties"""

import os 
import logging
import torch
import shutil
import numpy as np 

benchmark_name = {
    'blackscholes',
    'fft',
    'inversek2j',
    'jmeint',
    'jpeg',
    'kmeans',
    'sobel',
}

def benchmark_lr(benchmark):
    lr = {
        'blackscholes':1e-2,
        'fft':1e-4,
        'inversek2j':1e-4,
        'jmeint':1e-4,
        'jpeg':1e-4,
        'kmeans':1e-4,
        'sobel':1e-4,
    }
    return lr[benchmark]

def benchmark_input_output(benchmark):
    """
    [input_size, output_size]
    """
    input_output ={
        'blackscholes': [6,1],
        'fft': [1,2],
        'jpeg': [64,64],
        'kmeans': [6,1],
        'sobel': [9,1],
        'inversek2j': [2,2],
        'jmeint': [18,2]
    }
    return input_output[benchmark]

def error_func_name(benchmark):
    func = {
        'blackscholes': relative_error,
        'fft': fft_error,
        'jpeg': RMSE_error,
        'kmeans': relative_error,
        'sobel': Image_diff,
        'inversek2j': inversek2j_error,
        'jmeint': class_error
    }
    return func[benchmark]

def error_bound(benchmark):

    error_bound_grop_1 = {
        'blackscholes':0.1, 
        'fft':0.0001,  
        'inversek2j':0.02, 
        'jmeint':0.92,
        'jpeg':0.001,
        'kmeans':0.05,
        'sobel':0.05,
    }
    error_bound_grop_2 = {
        'blackscholes':0.15, 
        'fft':0.001,  
        'inversek2j':0.05, 
        'jmeint':0.92,
        'jpeg':0.005,
        'kmeans':0.1,
        'sobel':0.1,
    }
    return error_bound_grop_1[benchmark]

def accuracy(benchmark_name,origin,prediction):
    err_func = error_func_name(benchmark_name)
    funk = err_func
    if benchmark_name == 'jmeint':
        return funk(origin,prediction)
    else:
        err = funk(origin,prediction)
        err_bound = torch.tensor([error_bound(benchmark_name)]).to(torch.device("cuda"))
        return err_accuracy(err,err_bound)

def err_accuracy(err, error_bound):
    batch_size = err.size(0)
    res = []
    correct = err.le(error_bound.expand_as(err))
    correct_k = correct.view(-1).float().sum(0)
    res.append(correct_k.mul_(1.0 / batch_size))
    return res


def relative_error(origin,prediction):
    err = origin - prediction
    return torch.abs(torch.div(err,origin+1e-11))

def fft_error(origin,prediction):
    err = origin - prediction
    nominator = torch.sum(torch.pow(err,2),dim=1)
    denominator = torch.sum(torch.pow(origin,2),dim=1)
    denominator += torch.tensor([1e-21]).expand_as(denominator).to(torch.device("cuda"))
    return torch.div(nominator,denominator)

def inversek2j_error(origin,prediction):
    err = origin - prediction
    nominator = torch.sum(torch.pow(err,2),dim=1)
    denominator = torch.sum(torch.pow(origin,2),dim=1)
    denominator += torch.tensor([1e-21]).expand_as(denominator).to(torch.device("cuda"))
    return torch.div(nominator,denominator)

def class_error(origin,prediction):
    """ Computes the precision@k for the specified values of k """
    batch_size = origin.size(0)

    _, pred = prediction.topk(1, 1, True, True)
    pred = pred.t()
    # one-hot case
    if origin.ndimension() > 1:
        origin = origin.max(1)[1]

    correct = pred.eq(origin.view(1, -1).expand_as(pred))

    res = []
    correct_k = correct[:1].view(-1).float().sum(0)
    res.append(correct_k.mul_(1.0 / batch_size))
    
    return res

def Image_diff(origin,prediction):
    err = origin- prediction
    return torch.sqrt(torch.mean(torch.pow(err,2),dim=1))


def RMSE_error(origin,prediction):
    err = origin- prediction
    return torch.sqrt(torch.mean(torch.pow(err,2),dim=1))


def absolute_error(origin,prediction):
    return torch.abs(origin-prediction)


def get_logger(file_path):
    """ make python logger"""
    logger = logging.getLogger('auto_nn_ax')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format,datefmt='%m/%d %I:%M:%S %p')

    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    #stream_handler = logging.StreamHandler()
    #stream_handler.setFormatter(formatter)
    if(len(logger.handlers) == 0):
        logger.addHandler(file_handler)
    #logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

def param_size(model):
    """ Compute parameter size in kB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024.

class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def class_accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)


def as_markdown(args):
    """ Return args configs as markdown format """
    text = "|name|value|  \n|-|-|  \n"
    for attr, value in sorted(vars(args).items()):
        text += "|{}|{}|  \n".format(attr, value)

    return text

def print_params(args, prtf):
    prtf("")
    prtf("Parameters:")
    for attr, value in sorted(vars(args).items()):
        prtf("{}={}".format(attr.upper(), value))
    prtf("")


