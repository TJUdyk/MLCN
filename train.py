from ast import Param
import os
from statistics import mode
import tqdm
import time
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import detect_grad_nan, compute_accuracy, ensure_path, set_seed, setup_run
from common.losses import ContrastiveLoss , SupConLoss

from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.renet import RENet
from test import test_main, evaluate


def train(epoch, model, loader, optimizer, citerion_contrast ,citerion_spatial ,args=None):
    
    model.train()

    train_loader = loader['train_loader']
    train_loader_aux = loader['train_loader_aux']
    
    label = torch.arange(args.way).repeat(args.query).cuda() 

    loss_meter = Meter()
    acc_meter = Meter()

    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(train_loader)

    for i, ((data, train_labels), (data_aux, train_labels_aux)) in enumerate(zip(tqdm_gen, train_loader_aux), 1):

        data, train_labels = data.cuda(), train_labels.cuda() # torch.Size([80, 3, 84, 84])
        data_aux, train_labels_aux = data_aux.cuda(), train_labels_aux.cuda() # torch.Size([64, 3, 84, 84])

        # Forward images (3, 84, 84) -> (C, H, W)   
        model.module.mode = 'encoder' # ResNet12
        data = model(data)  
        data_aux = model(data_aux) 

        model.module.mode = 'cca'
        data_shot, data_query = data[:k], data[k:]
        logits, absolute_logits = model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_query))
        # 原型之间的loss
        loss_pro = F.cross_entropy(logits, label) # CCA的
        absolute_loss = F.cross_entropy(absolute_logits, train_labels[k:])

        model.module.mode = 'fc'
        logits_aux = model(data_aux)  
        loss_aux = F.cross_entropy(logits_aux, train_labels_aux)
        loss_aux = loss_aux + absolute_loss # FC和ResNet以及CCA的分类的loss
        #loss = loss_aux + loss_pro * args.lamb
        loss = loss_aux
        acc = compute_accuracy(logits, label)
        
        loss_meter.update(loss.item())
        acc_meter.update(acc)
        tqdm_gen.set_description(f'[train] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        detect_grad_nan(model)
        optimizer.step()
        optimizer.zero_grad()

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()
   
def train_main(args):
    Dataset = dataset_builder(args)

    trainset = Dataset('train', args)
    args.trainset_len = len(trainset) #  100
    train_sampler = CategoriesSampler(trainset.label, len(trainset.data) // args.batch, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)

    trainset_aux = Dataset('train', args)
    train_loader_aux = DataLoader(dataset=trainset_aux, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    train_loaders = {'train_loader': train_loader, 'train_loader_aux': train_loader_aux}

    valset = Dataset('val', args)
    args.valset_len = len(valset)
    val_sampler = CategoriesSampler(valset.label, args.val_episode, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)
    ''' fix val set for all epochs '''
    val_loader = [x for x in val_loader]
    #val_lodaer_clu = val_loader

    set_seed(args.seed)
    model = RENet(args).cuda()
    model = nn.DataParallel(model, device_ids=args.device_ids)        

    if not args.no_wandb:
        wandb.watch(model)
    print(model)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    #eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch, eta_min, -1)
    
    max_acc, max_epoch = 0.0, 0
    set_seed(args.seed)
    citerion_spatial = ContrastiveLoss(temperature=args.temperature)
    citerion_contrast = SupConLoss(temperature=args.temperature) 
    for epoch in range(1, args.max_epoch + 1):
        start_time = time.time()
        
        train_loss, train_acc, _ = train(epoch, model, train_loaders, optimizer, citerion_contrast ,citerion_spatial , args)
        
        val_loss ,val_acc, _ = evaluate(epoch, model, val_loader, args, set='val')
        
        if not args.no_wandb:
            wandb.log({'train/loss': train_loss, 'train/acc': train_acc, 'val/loss': val_loss, 'val/acc': val_acc}, step=epoch)
        
        if val_acc > max_acc:
            #sys.stdout = Logger('./logger/a.log', sys.stdout)
            print(f'[ log ] *********A better model is found ({val_acc:.3f}) *********')
            max_acc, max_epoch = val_acc, epoch
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, 'max_acc.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_max_acc.pth'))
        
        if args.save_all:
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, f'epoch_{epoch}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, f'optimizer_epoch_{epoch}.pth'))

        epoch_time = time.time() - start_time
        local_time = time.asctime(time.localtime(time.time()))
        #sys.stdout = Logger('./logger/a.log', sys.stdout)
        print(f'[ log ] saving @ {args.save_path}')
        print(f'[ log ] roughly {(args.max_epoch - epoch) / 3600. * epoch_time:.2f} h left')
        print(f'[ log ] time: @{local_time}\n')
        
        lr_scheduler.step()
    to_save ={
            'parms': model.state_dict(),
            'epoch': epoch,
            'loss' : train_loss,
            'optimizer':optimizer.state_dict()
            }
    #torch.save(to_save ,os.path.join(args.save_path, 'save_all.pth') )
    return model

if __name__ == '__main__':
     
    args = setup_run(arg_mode='train')
    model = train_main(args)

    # wandb的代码
    test_acc, test_ci = test_main(model, args)
    if not args.no_wandb:
        wandb.log({'test/acc': test_acc, 'test/confidence_interval': test_ci})
