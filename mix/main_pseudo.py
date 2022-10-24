#!/usr/bin/env python
from __future__ import division
from operator import __or__
import shutil
import torch.backends.cudnn as cudnn
import argparse
import matplotlib as mpl
mpl.use('Agg')
import models

import sys
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
from scipy.stats import entropy
from collections import OrderedDict
from load_data  import *

model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains Preresnet/Resnext on CIFAR10', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10', 'mnist', 'tiny-imagenet-200'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--data_dir', type = str, default = 'cifar10',
                        help='file where results are to be written')
parser.add_argument('--root_dir', type = str, default = 'experiments',
                        help='folder where results are to be stored')
parser.add_argument('--labels_per_class', type=int, default=5000, metavar='NL',
                    help='labels_per_class')
parser.add_argument('--unlabels_per_class', type=int, default=5000, metavar='NL',
                    help='unlabels_per_class')
parser.add_argument('--arch', metavar='ARCH', default='resnext29_8_64', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
parser.add_argument('--initial_channels', type=int, default=64, choices=(16,64))
# Optimization options
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--train', type=str, default = 'vanilla_pseudo', choices =['vanilla_pseudo','mixup_pseudo', 'mixup_hidden_pseudo','partial_mixup_pseudo','partial_mixup_hidden_pseudo','stages_mixup_pseudo','stages_mixup_hidden_pseudo'])
parser.add_argument('--uda_loss',action='store_true',default=False,help='inception score loss')
#parser.add_argument('--pseudo',action='store_true',default=False,help='self training')
parser.add_argument('--mixup_alpha', type=float, default=0.0, help='alpha parameter for mixup')
parser.add_argument('--r', type=float, default=0.3, help='Training to add inception loss')
parser.add_argument('--p', type=float, default=0.6, help='Training to divide stage1 with stage2')
parser.add_argument('--q', type=float, default=0.9, help='Training to divide stage2 with stage3')
parser.add_argument('--dropout', action='store_true', default=False,
                    help='whether to use dropout or not in final layer')
#parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--data_aug', type=int, default=1)
parser.add_argument('--decay', type=float, default=0.0001, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
# attack
parser.add_argument('--attack',action='store_true',default=False,help='attack training')
parser.add_argument('--partial_attack',action='store_true',default=False,help='partial mix with attack training')
parser.add_argument('--pgd_eps',type=float, default=8/255)
parser.add_argument('--pgd_alpha',type=float,default=2/255)
parser.add_argument('--pgd_step_size',type=int,default=7)

parser.add_argument('--noise', type=float, default=0.0)

# ssl
parser.add_argument('--threshold', type=str, default = 'exp_schedule', choices =["", "linear_schedule", "log_schedule", "exp_schedule"],help="anneal schedule of training signal annealing.")
parser.add_argument('--thre_start', type=float, default=0.9)
parser.add_argument('--thre_end', type=float, default=0.95)

args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

out_str = str(args)
#print(out_str)


"""
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
"""
cudnn.benchmark = True

def copy_script_to_folder(caller_path, folder):
    script_filename = caller_path.split('/')[-1]
    script_relative_path = os.path.join(folder, script_filename)
    # Copying script
    shutil.copy(caller_path, script_relative_path)

def experiment_name_non_mnist(dataset='cifar10',arch='preresnet18',train = 'vanilla_pseudo'):
    exp_name = dataset
    exp_name += '_' + str(arch)+'_'
    exp_name += str(train)
    if args.attack :
        exp_name += '_attack'
    if args.partial_attack :
        exp_name += '_partial_attack'
    if args.uda_loss:
        exp_name += '_uda_loss'
   # if args.pseudo :
   #     exp_name += '_pseudo'
    exp_name += '_thre'+str(args.thre_start)+'_'+str(args.thre_end)
    print('experiement name: ' + exp_name)
    return exp_name

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_threshold(threshold, global_step, num_train_steps, start, end):
  step_ratio =float(global_step) / float(num_train_steps)
  if threshold == "linear_schedule":
    coeff = step_ratio
  elif threshold == "exp_schedule":
    scale = 5
    # [exp(-5), exp(0)] = [1e-2, 1]
    coeff = np.exp((step_ratio - 1) * scale)
  elif threshold == "log_schedule":
    scale = 5
    # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
    coeff = 1 - np.exp((-step_ratio) * scale)
  return coeff * (end - start) + start

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def inception_score(preds,splits):
    split_scores = []
    N= len(preds)
    preds = preds.detach().cpu().numpy()
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(-np.mean(scores)))
        return Variable(torch.mean(torch.Tensor(split_scores)), requires_grad=True).cuda()

bce_loss = nn.BCELoss().cuda()
softmax = nn.Softmax(dim=1).cuda()

def train(train1_loader,train2_loader,model,optimizer, epoch, args,log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()

    iter1=iter(train1_loader)
    iter2=iter(train2_loader)

    for i in range(len(train1_loader)):
        try:
            input_, target = next(iter1)
            input_2, target_2 = next(iter2)
        except StopIteration:
            iter2= iter(train2_loader)
        #print(i,len(train1_loader))
        target,target_2= target.long(),target_2.long()
        input_, input_2,target,target_2 = input_.cuda(), input_2.cuda(),target.cuda(),  target_2.cuda()
        input_var,input_2_var, target_var ,target_2_var= Variable(input_), Variable(input_2),Variable(target),Variable(target_2)
        
        data_time.update(time.time() - end)

        if args.train == 'vanilla_pseudo':
            output, reweighted_target = model(x=input_var,target= target_var, noise=args.noise)

        elif args.train == 'mixup_pseudo':
            output, reweighted_target = model(x=input_var,target= target_var, mixup=True, mixup_alpha=args.mixup_alpha,noise=args.noise)

        elif args.train == 'mixup_hidden_pseudo':
            output, reweighted_target = model(x=input_var,target= target_var, mixup_hidden=True, mixup_alpha=args.mixup_alpha,noise=args.noise)

        elif args.train == 'partial_mixup_pseudo':
            output, reweighted_target = model(x=input_var,xb=input_2_var,target= target_var, targetb=target_2_var,mixup=True, mixup_alpha=args.mixup_alpha,noise=args.noise)

        elif args.train == 'partial_mixup_hidden_pseudo':
            output, reweighted_target = model(x=input_var,xb=input_2_var,target= target_var, targetb=target_2_var,mixup_hidden=True, mixup_alpha=args.mixup_alpha,noise=args.noise)

        elif args.train == 'stages_mixup_pseudo':
            if epoch >= args.epochs * args.q:
                mask = random.random()
                threshold = (args.epochs - epoch) / (args.epochs - args.epochs * args.q)
                if mask < threshold:
                    output, reweighted_target = model(x=input_var,target= target_var, mixup=True, mixup_alpha=args.mixup_alpha,noise=args.noise)
                   # output, reweighted_target = model(x=input_var,xb=input_2_var,target= target_var, targetb=target_2_var,mixup=True, mixup_alpha=args.mixup_alpha,noise=args.noise)
                else:
                    output, reweighted_target = model(x=input_var,xb=input_2_var,target= target_var, targetb=target_2_var,mixup=True, mixup_alpha=args.mixup_alpha,noise=args.noise)
                   # output, reweighted_target = model(x=input_var,target= target_var, mixup=True, mixup_alpha=args.mixup_alpha,noise=args.noise)
            elif epoch >= args.epochs * args.p:
                if epoch % 2 == 0:
                    output, reweighted_target = model(x=input_var,target= target_var, mixup=True, mixup_alpha=args.mixup_alpha,noise=args.noise)
                   # output, reweighted_target = model(x=input_var,xb=input_2_var,target= target_var, targetb=target_2_var,mixup=True, mixup_alpha=args.mixup_alpha,noise=args.noise)

                else:
                    output, reweighted_target = model(x=input_var,xb=input_2_var,target= target_var, targetb=target_2_var,mixup=True, mixup_alpha=args.mixup_alpha,noise=args.noise)
                    #output, reweighted_target = model(x=input_var,target= target_var, mixup=True, mixup_alpha=args.mixup_alpha,noise=args.noise)
            else:
                output, reweighted_target = model(x=input_var,target= target_var, mixup=True, mixup_alpha=args.mixup_alpha,noise=args.noise)
               # output, reweighted_target = model(x=input_var,xb=input_2_var,target= target_var, targetb=target_2_var,mixup=True, mixup_alpha=args.mixup_alpha,noise=args.noise)
        elif args.train == 'stages_mixup_hidden_pseudo':
            if epoch >= args.epochs * args.q:
                mask = random.random()
                threshold = (args.epochs - epoch) / (args.epochs - args.epochs * args.q)
                if mask < threshold:
                    output, reweighted_target = model(x=input_var,target= target_var, mixup_hidden=True, mixup_alpha=args.mixup_alpha,noise=args.noise)                   # output, reweighted_target = model(x=input_var,xb=input_2_var,target= target_var, targetb=target_2_var,mixup=True, mixup_alpha=args.mixup_alpha,noise=args.noise)
                else:
                    output, reweighted_target = model(x=input_var,xb=input_2_var,target= target_var, targetb=target_2_var,mixup_hidden=True, mixup_alpha=args.mixup_alpha,noise=args.noise)
                   # output, reweighted_target = model(x=input_var,target= target_var, mixup=True, mixup_alpha=args.mixup_alpha,noise=args.noise)
            elif epoch >= args.epochs * args.p:
                if epoch % 2 == 0:
                    output, reweighted_target = model(x=input_var,target= target_var, mixup_hidden=True, mixup_alpha=args.mixup_alpha,noise=args.noise)                   # output, reweighted_target = model(x=input_var,xb=input_2_var,target= target_var, targetb=target_2_var,mixup=True, mixup_alpha=args.mixup_alpha,noise=args.noise)

                else:
                    output, reweighted_target = model(x=input_var,xb=input_2_var,target= target_var, targetb=target_2_var,mixup_hidden=True, mixup_alpha=args.mixup_alpha,noise=args.noise)
                    #output, reweighted_target = model(x=input_var,target= target_var, mixup=True, mixup_alpha=args.mixup_alpha,noise=args.noise)
            else:
                output, reweighted_target = model(x=input_var,target= target_var, mixup_hidden=True, mixup_alpha=args.mixup_alpha,noise=args.noise)
               # output, reweighted_target = model(x=input_var,xb=input_2_var,target= target_var, targetb=target_2_var,mixup=True, mixup_alpha=args.mixup_alpha,noise=args.noise)
 
#        elif args.train == 'stages_mixup_hidden_pseudo':
#            if epoch >= args.epochs * args.q:
#                mask = random.random()
#                threshold = (args.epochs - epoch) / (args.epochs - args.epochs * args.q)
#                if mask < threshold:
#                    output, reweighted_target = model(x=input_var,xb=input_2_var,target= target_var, targetb=target_2_var,mixup_hidden=True, mixup_alpha=args.mixup_alpha,noise=args.noise)
#                else:
#                    output, reweighted_target = model(x=input_var,target= target_var, mixup_hidden=True, mixup_alpha=args.mixup_alpha,noise=args.noise)
#            elif epoch >= args.epochs * args.p:
#                if epoch % 2 == 0:
#                    output, reweighted_target = model(x=input_var,xb=input_2_var,target= target_var, targetb=target_2_var,mixup_hidden=True, mixup_alpha=args.mixup_alpha,noise=args.noise)

#                else:
#                    output, reweighted_target = model(x=input_var,target= target_var, mixup_hidden=True, mixup_alpha=args.mixup_alpha,noise=args.noise)
#            else:
#                output, reweighted_target = model(x=input_var,xb=input_2_var,target= target_var, targetb=target_2_var,mixup_hidden=True, mixup_alpha=args.mixup_alpha,noise=args.noise)
        else:
            assert False

        loss = bce_loss(softmax(output), reweighted_target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input_.size(0))
        top1.update(prec1.item(), input_.size(0))
        top5.update(prec5.item(), input_.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, i, len(train1_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)

    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                              error1=100 - top1.avg),
        log)
    return top1.avg, top5.avg, losses.avg


def validate(epoch, val_loader, model, mode, log):
    print_log("\n[Epoch {}] Start validation ({})...".format(epoch, mode), log)
    criterion= nn.CrossEntropyLoss().cuda()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    for i, (input_, target) in enumerate(val_loader):
        target = target.cuda()
        input_ = input_.cuda()
        with torch.no_grad():
            input_var = Variable(input_)
            target_var = Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input_.size(0))
        top1.update(prec1.item(), input_.size(0))
        top5.update(prec5.item(), input_.size(0))

    print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f} Loss: {losses.avg:.3f} '.format(top1=top1, top5=top5, error1=100-top1.avg, losses=losses), log)

    return top1.avg, losses.avg


best_acc = 0
def main():
    exp_name=experiment_name_non_mnist(dataset=args.dataset,arch=args.arch,train = args.train)
    exp_dir = args.root_dir+exp_name

    if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

    copy_script_to_folder(os.path.abspath(__file__), exp_dir)
    result_png_path = os.path.join(exp_dir, 'results.png')

    global best_acc

    log = open(os.path.join(exp_dir, 'log.txt'.format(args.manualSeed)), 'w')
    #print_log('save path : {}'.format(exp_dir), log)

    per_img_std = False
    stride = 1

    raw_train_data, test_data, num_classes = load_raw_dataset(args.data_aug, args.dataset, args.data_dir)

    #print_log("=> creating model '{}'".format(args.arch), log)
    net = models.__dict__[args.arch](num_classes,args.dropout,per_img_std, stride).cuda()
    #print_log("=> network :\n {}".format(net), log)
    args.num_classes = num_classes
    optimizer = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=args.momentum,
                                weight_decay=args.decay, nesterov=True)
    recorder = RecorderMeter(args.epochs)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = recorder.max_accuracy(False)
            print_log("=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(args.resume, best_acc, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    train_loss = []
    train_acc=[]
    test_loss=[]
    test_acc=[]
    attack_aft_acc = []
    attack_aft_loss = []
    
    for epoch in range(args.start_epoch, args.epochs):
        adv_test_dataset=attack_test_data(test_data,net,batch_size=512,num_iter=args.pgd_step_size,eps=args.pgd_eps,alpha=args.pgd_alpha)
        adv_test_dataset.inputs = adv_test_dataset.inputs.cpu()

        labeled_train_loader, unlabeled_train_loader,all_train_loader= load_sub_trainset(raw_train_data, num_classes, args.batch_size, args.workers,
                                                 args.labels_per_class,args.unlabels_per_class)
#        threshold = get_threshold(args.threshold, epoch, args.epochs, args.thre_start, args.thre_end)
#        pseudo_dataset = pseudo_data(unlabeled_train_loader, net,args.noise, threshold)
#        pseudo_loader = torch.utils.data.DataLoader(pseudo_dataset, batch_size=args.batch_size,shuffle=False,num_workers=args.workers, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True)
        adv_test_loader = torch.utils.data.DataLoader(adv_test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)


        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs,
                                                                                   need_time, current_learning_rate) \
            + (' [Best : Test Accuracy={:.2f}, '
               'Adv Test Accuracy={:.2f}]'.format(recorder.max_accuracy(False),recorder.max_at_aft_acc())), log)

        # train for one epoch
 #       if args.train == 'vanilla_pseudo' or args.train == 'mixup_pseudo' or args.train == 'mixup_hidden_pseudo':
 #           tr_acc, tr_acc5, tr_los = train(labeled_train_loader, labeled_train_loader,net, optimizer, epoch, args,log)
 #           if len(pseudo_loader) >0:
 #               tr_acc, tr_acc5, tr_los = train(pseudo_loader, pseudo_loader, net, optimizer, epoch, args, log)
 #       else:
 #           tr_acc, tr_acc5, tr_los = train(labeled_train_loader, labeled_train_loader, net, optimizer, epoch, args,log)
 #           if len(pseudo_loader) >0:
 #               tr_acc, tr_acc5, tr_los = train(pseudo_loader, labeled_train_loader, net, optimizer, epoch, args,log)
 #       val_acc, val_los = validate(epoch, test_loader, net, "Test", log)
 #       at_aft_acc, at_aft_loss = validate(epoch, adv_test_loader, net, "Adversarial attack", log)

        tr_acc, tr_acc5, tr_los = train(labeled_train_loader, labeled_train_loader,net, optimizer, epoch, args, log)
        val_acc, val_los = validate(epoch, test_loader, net, "Test", log)
        thre = get_threshold(args.threshold, epoch, args.epochs, args.thre_start, args.thre_end)
        if val_acc/100.0 > thre:
            pseudo_dataset = pseudo_data(unlabeled_train_loader, net, args.noise, 0.0)
            pseudo_loader = torch.utils.data.DataLoader(pseudo_dataset, batch_size=args.batch_size, shuffle=True,
                                                        num_workers=args.workers, pin_memory=True)
            if args.train == 'vanilla_pseudo':
                tr2_acc, tr2_acc5, tr2_los = train(pseudo_loader, pseudo_loader,net, optimizer, epoch, args, log)
            elif args.train == 'mixup_pseudo':
                tr2_acc, tr2_acc5, tr2_los = train(pseudo_loader, pseudo_loader,net, optimizer, epoch, args, log)
            elif args.train == 'mixup_hidden_pseudo':
                tr2_acc, tr2_acc5, tr2_los = train(pseudo_loader, pseudo_loader,net, optimizer, epoch, args, log)
            else:
                tr2_acc, tr2_acc5, tr2_los = train(pseudo_loader, labeled_train_loader,net, optimizer, epoch, args, log)
        at_aft_acc, at_aft_loss = validate(epoch, adv_test_loader, net, "Adversarial attack", log)

        train_loss.append(tr_los)
        train_acc.append(tr_acc)
        test_loss.append(val_los)
        test_acc.append(val_acc)
        attack_aft_loss.append(at_aft_loss)
        attack_aft_acc.append(at_aft_acc)

        recorder.update(epoch, tr_los, tr_acc, val_los, val_acc,at_aft_loss,at_aft_acc)


        is_best = False
        if val_acc > best_acc:
            is_best = True
            best_acc = val_acc

        save_checkpoint({
          'epoch': epoch + 1,
          'arch': args.arch,
          'state_dict': net.state_dict(),
          'recorder': recorder,
          'optimizer' : optimizer.state_dict(),
        }, is_best, exp_dir, 'checkpoint.pth.tar')

        train_log = OrderedDict()
        train_log['train_loss'] = train_loss
        train_log['train_acc']=train_acc
        train_log['test_loss']=test_loss
        train_log['test_acc']=test_acc
        train_log['attack_after_training_loss'] = attack_aft_loss
        train_log['attack_after_training_acc'] = attack_aft_acc

        pickle.dump(train_log, open( os.path.join(exp_dir,'log.pkl'), 'wb'))
        plotting(exp_dir)

        if epoch == args.epochs:
            break

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(result_png_path)

    print_log("Saving final model...", log)
    torch.save(net.state_dict(), exp_dir + '/final_model.pth')
    print_log("\nfinish", log)
    
    log.close()


if __name__ == '__main__':
    main()
