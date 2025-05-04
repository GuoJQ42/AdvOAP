from utils import progress_bar
from loss.trades import trades_loss
from loss.tradition_loss import tradition_loss_regression, tradition_loss_two_class_classification
from loss.adv_oap_loss import adv_oap


import torch
import torch.optim as optim
import random


def Adv_Trainer(args, model, device, train_loader, optimizer, epochs, scheduler):
    model.train()
    train_clean_loss = 0
    train_adv_loss = 0
    correct_clean = 0
    correct_adv = 0
    num_example = 0 
    print('Epoch: %d'%epochs)

    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        optimized_loss, batch_metrics = trades_loss(model, data, targets, step_size = args['step_size'], epsilon = args['perturbation_size'], perturb_steps = args['perturb_steps'], beta = args['beta'], attack = args['attack_type'])
        optimizer.zero_grad()
        optimized_loss.backward()
        optimizer.step()

        num_example += targets.size(0)
        train_clean_loss += batch_metrics['clean_loss_avg']
        train_adv_loss += batch_metrics['adv_loss']
        correct_clean += batch_metrics['clean_acc_num_avg']
        correct_adv += batch_metrics['adv_acc_num_avg']

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Adv Loss: %.3f | Acc: %.3f%% (%d/%d) | AdvAcc: %.3f%% (%d/%d)' % (train_clean_loss/(batch_idx+1), train_adv_loss/(batch_idx+1), 100.*correct_clean/num_example, correct_clean, num_example, 100.*correct_adv/num_example, correct_adv, num_example))

        # with open('./output/result.txt', 'a') as f:
        #     f.write('Epoch: %d, Batch: %d '%(epochs, batch_idx))
        #     f.write('Loss: %.3f | Adv Loss: %.3f | Acc: %.3f%% (%d/%d) | AdvAcc: %.3f%% (%d/%d)\n' % (train_clean_loss/(batch_idx+1), train_adv_loss/(batch_idx+1), 100.*correct_clean/num_example, correct_clean, num_example, 100.*correct_adv/num_example, correct_adv, num_example))

    scheduler.step()



def Adv_Ensemble_Trainer(train_args, ensemble_model, device, train_loader, optimizer, epoch_idx, scheduler):
    '''
    ensemble_model (nn.Module): the ensemble model. Use .models to obtain all the sub-models. The last model is the one to be optimized.
    '''
    ensemble_model.train()
    
    train_clean_loss = 0
    train_adv_loss = 0
    train_clean_acc_num = 0
    train_adv_acc_num = 0

    train_reg = 0

    num_example = 0

    grad_div = 0
    cross_div = 0
    avg_loss = 0
    ens_loss = 0

    print('Epoch: %d'%(epoch_idx))
    for batch_idx, (x, y) in enumerate(train_loader):    

        x, y = x.to(device), y.to(device)


        if train_args['method'][0:7] == 'adv_oap':
            optimized_loss, batch_metrics = adv_oap(ensemble_model, x, y, device, step_size = train_args['step_size'], epsilon=train_args['perturbation_size'], perturb_steps=train_args['perturb_steps'], attack_type=train_args['attack_type'], alpha=train_args['alpha'], beta=train_args['beta'], num_classes=train_args['num_classes'])
        elif train_args['method'][0:26] == 'Two_class_train_regression':
            optimized_loss, batch_metrics = tradition_loss_regression(ensemble_model, x, y)
        elif train_args['method'][0:30] == 'Two_class_train_classification':
            optimized_loss, batch_metrics = tradition_loss_two_class_classification(ensemble_model, x, y)
        
        optimizer.zero_grad()
        optimized_loss.backward()
        optimizer.step()

        num_example += y.size(0)
        train_clean_loss += batch_metrics['clean_loss']
        train_adv_loss += batch_metrics['adv_loss']
        train_clean_acc_num += batch_metrics['clean_acc_num']
        train_adv_acc_num += batch_metrics['adv_acc_num']
        train_reg += batch_metrics['reg']

        if 'grad_div' in batch_metrics:
            grad_div += batch_metrics['grad_div']
        
        if 'cross_div' in batch_metrics:
            cross_div += batch_metrics['cross_div']

        if 'avg_loss' in batch_metrics:
            avg_loss += batch_metrics['avg_loss']
        
        if 'ens_loss' in batch_metrics:
            ens_loss += batch_metrics['ens_loss']

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f | Adv Loss: %.3f | Adv Acc: %.3f | reg: %.3e | GradDiv: %.3f | CrossDiv: %.3f' % (train_clean_loss/num_example, train_clean_acc_num/num_example, train_adv_loss/num_example, train_adv_acc_num/num_example, train_reg/num_example, grad_div/num_example, cross_div/num_example))




    with open('./output/result_train_%s_%s.txt'%(train_args['model'], train_args['method']), 'a') as f:
        f.write('Epoch: %d, Batch: %d \n'%(epoch_idx, batch_idx))
        f.write('Loss: %.3f | Acc: %.3f | Adv Loss: %.3f | Adv Acc: %.3f | reg: %.3e\n' % (train_clean_loss/num_example, train_clean_acc_num/num_example, train_adv_loss/num_example, train_adv_acc_num/num_example, train_reg/num_example))
    
    

    if not (scheduler is None):
        scheduler.step()
    
    ensemble_model.eval()



