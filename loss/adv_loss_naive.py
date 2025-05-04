import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import random
# from trades import trades_loss

from .metrics import accuracy, accuracy_regression
from attacks.pgd import PGD_attack, two_stage_PGD, PGD_attack_regression

def adv_loss_naive_submodel(model, x_natural, y, step_size=0.002, epsilon=0.031, perturb_steps=40,  attack='linf-pgd', keep_clean = False):
    """
    Adversarial training.
    """

    # define KL-loss
    loss_func = torch.nn.CrossEntropyLoss()
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example

    if attack == 'linf-pgd':
        x_adv = PGD_attack(model, x_natural, y, epsilon, perturb_steps, step_size)
    
    elif attack == 'l2-pgd':
        pass
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')
    model.train()

    # calculate robust loss
    logits_adv = model.models[-1](x_adv)


    loss_robust = loss_func(logits_adv, y)
    logits_natural = model.models[-1](x_natural)
    loss_natural = loss_func(logits_natural, y)
    batch_metrics = {'clean_loss': loss_natural.item(), 'adv_loss': loss_robust.item(), 'clean_acc_num': accuracy(y, logits_natural.detach())*float(y.size(0)), 'adv_acc_num': accuracy(y, logits_adv.detach())*float(y.size(0))}
    if keep_clean:
        return loss_natural + loss_robust, batch_metrics
    else:
        return loss_robust, batch_metrics



def adv_loss_naive(model, x_natural, y, step_size=0.002, epsilon=0.031, perturb_steps=40, attack_type='linf-pgd', keep_clean = False, if_regression = False):
    """
    Adversarial training.
    """
    # tmp_y = torch.ones([x_natural.size()[0],10])/10
    # tmp_y = tmp_y.cuda()
    tmp_y = y

    # define KL-loss
    if if_regression:
        loss_func = torch.nn.MSELoss()
    else:
        loss_func = torch.nn.CrossEntropyLoss()
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example

    if attack_type == 'l_infty' or attack_type == 'l_2':
        if if_regression:
            # breakpoint()
            x_adv = PGD_attack_regression(model, x_natural, tmp_y, epsilon, perturb_steps, step_size, type=attack_type)
        else:
            x_adv = PGD_attack(model, x_natural, tmp_y, epsilon, perturb_steps, step_size, type=attack_type)
    else:
        raise ValueError(f'Attack={attack_type} not supported for TRADES training!')
    model.train()

    # calculate robust loss
    if if_regression:
        logits_adv = model(x_adv)
        loss_robust = loss_func(logits_adv.flatten(), tmp_y)
        logits_natural = model(x_natural)
        loss_natural = loss_func(logits_natural.flatten(), tmp_y)

        batch_metrics = {'clean_loss': loss_natural.item(), 'adv_loss': loss_robust.item(), 'clean_acc_num': accuracy_regression(y, logits_natural.detach())*float(y.size(0)), 'adv_acc_num': accuracy_regression(y, logits_adv.detach())*float(y.size(0))}
    else:
        logits_adv = model(x_adv)
        loss_robust = loss_func(logits_adv, tmp_y)
        logits_natural = model(x_natural)
        loss_natural = loss_func(logits_natural, tmp_y)

        batch_metrics = {'clean_loss': loss_natural.item(), 'adv_loss': loss_robust.item(), 'clean_acc_num': accuracy(y, logits_natural.detach())*float(y.size(0)), 'adv_acc_num': accuracy(y, logits_adv.detach())*float(y.size(0))}

    batch_metrics['reg'] = 0

    if keep_clean:
        return loss_natural + loss_robust, batch_metrics
    else:
        # return loss_natural, batch_metrics
        return loss_robust, batch_metrics



def adv_loss_naive_multi(ensemble_model, x_natural, y, step_size=0.002, epsilon=0.031, perturb_steps=40,  attack_type='l_infty', keep_clean = False):
    """
    Adversarial training.
    """

    # define KL-loss
    loss_func = torch.nn.CrossEntropyLoss()
    
    final_loss = 0
    batch_metrics = {}
    # generate adversarial example

    adv_idx = random.randint(0, len(ensemble_model.models)-1)

    adv_average = x_natural

    for model_idx, model in enumerate(ensemble_model.models):
        model.eval()
        if attack_type == 'l_infty' or attack_type == 'l_2':
            x_adv = PGD_attack(model, x_natural, y, epsilon, perturb_steps, step_size, type=attack_type)
        else:
            raise ValueError(f'Attack={attack_type} not supported for TRADES training!')
        
        if model_idx == adv_idx:
            final_adv = x_adv
        
        adv_average = adv_average + x_adv

        model.train()
        logits_adv = model(x_adv)
        loss_robust = loss_func(logits_adv, y)
        logits_natural = model(x_natural)
        loss_natural = loss_func(logits_natural, y)
        
        if len(batch_metrics) == 0:
            batch_metrics = {'clean_loss': loss_natural.item(), 'adv_loss': loss_robust.item(), 'clean_acc_num': accuracy(y, logits_natural.detach())*float(y.size(0)), 'adversarial_acc_num': accuracy(y, logits_adv.detach())*float(y.size(0))}
        else:
            tmp_batch_metrics = {'clean_loss': loss_natural.item(), 'adv_loss': loss_robust.item(), 'clean_acc_num': accuracy(y, logits_natural.detach())*float(y.size(0)), 'adversarial_acc_num': accuracy(y, logits_adv.detach())*float(y.size(0))}
            for key in batch_metrics.keys():
                batch_metrics[key] = batch_metrics[key] + tmp_batch_metrics[key]
        if keep_clean:
            final_loss = final_loss + loss_natural + loss_robust
        else:
            final_loss = final_loss + loss_robust
        

    for key in batch_metrics.keys():
        batch_metrics[key] = batch_metrics[key] / len(ensemble_model.models)
    
    final_loss = final_loss

    return final_loss, final_adv, batch_metrics


def adv_loss_two_stage(ensemble_model, x_natural, y, step_size=0.002, epsilon=0.031, perturb_steps=40,  attack_type='l_infty', keep_clean = False):
    """
    Adversarial training.
    """

    # define KL-loss
    loss_func = torch.nn.CrossEntropyLoss()
    
    final_loss = 0
    batch_metrics = {}

    x_adv_average, x_advs = two_stage_PGD(ensemble_model, x_natural, y, epsilon, perturb_steps, step_size, type=attack_type)

    for idx, model in enumerate(ensemble_model.models):
        
        model.train()
        logits_adv = model(x_advs[idx])
        loss_robust = loss_func(logits_adv, y)

        final_loss = final_loss + loss_robust

        logits_adv = model(x_adv_average)
        loss_robust = loss_func(logits_adv, y)

        final_loss = final_loss + loss_robust

        logits_natural = model(x_natural)
        loss_natural = loss_func(logits_natural, y)
        
        if len(batch_metrics) == 0:
            batch_metrics = {'clean_loss_avg': loss_natural.item(), 'adv_loss': loss_robust.item(), 'clean_acc_num_avg': accuracy(y, logits_natural.detach())*float(y.size(0)), 'adv_acc_num_avg': accuracy(y, logits_adv.detach())*float(y.size(0))}
        else:
            tmp_batch_metrics = {'clean_loss_avg': loss_natural.item(), 'adv_loss': loss_robust.item(), 'clean_acc_num_avg': accuracy(y, logits_natural.detach())*float(y.size(0)), 'adv_acc_num_avg': accuracy(y, logits_adv.detach())*float(y.size(0))}
            for key in batch_metrics.keys():
                batch_metrics[key] = batch_metrics[key] + tmp_batch_metrics[key]

        

    for key in batch_metrics.keys():
        batch_metrics[key] = batch_metrics[key] / len(ensemble_model.models)
    
    final_loss = final_loss/len(ensemble_model.models)/2

    return final_loss, batch_metrics


def adv_loss_resiual_sum(ensemble_model, x_natural, y, step_size=0.002, epsilon=0.031, perturb_steps=40,  attack_type='l_infty', keep_clean = False):
    """
    Adversarial training.
    """

    # define KL-loss
    loss_func = torch.nn.CrossEntropyLoss()
    
    final_loss = 0
    logits_adv_average = 0
    prod_adv_average = 0
    resiual_sum = 0
    batch_metrics = {}

    x_advs = []
    
    for model in ensemble_model.models:
        x_adv = PGD_attack(model, x_natural, y, epsilon, perturb_steps, step_size, attack_type, bias=None)
        x_advs.append(x_adv)


    for idx, model in enumerate(ensemble_model.models):
        model.train()
        logits_adv = model(x_advs[idx])
        loss_robust = loss_func(logits_adv, y)
        
        logits_natural = model(x_natural)
        loss_natural = loss_func(logits_natural, y)

        logits_adv_average += logits_adv
        # prod_adv_average += F.softmax(logits_adv, dim=1)

        # resiual += (logits_adv - logits_natural)
        # # final_loss += loss_natural
        final_loss += loss_robust
        # final_loss = final_loss + loss_robust

        # alpha = 0.2
        # breakpoint()
        resiual = (F.softmax(logits_adv, dim=1) - F.softmax(logits_natural, dim=1))
        # final_loss += alpha*torch.norm(resiual, dim=1).mean()

        if idx == 0:
            logits_pre = F.softmax(logits_adv, dim=1)
        else:
            logits_diff = torch.norm(F.softmax(logits_adv, dim=1) - logits_pre, dim=1).mean().item()
        
        resiual_sum += resiual
        
        if len(batch_metrics) == 0:
            batch_metrics = {'clean_loss_avg': loss_natural.item(), 'adv_loss': loss_robust.item(), 'clean_acc_num_avg': accuracy(y, logits_natural.detach())*float(y.size(0)), 'adv_acc_num_avg': accuracy(y, logits_adv.detach())*float(y.size(0))}
        else:
            tmp_batch_metrics = {'clean_loss_avg': loss_natural.item(), 'adv_loss': loss_robust.item(), 'clean_acc_num_avg': accuracy(y, logits_natural.detach())*float(y.size(0)), 'adv_acc_num_avg': accuracy(y, logits_adv.detach())*float(y.size(0))}
            for key in batch_metrics.keys():
                batch_metrics[key] = batch_metrics[key] + tmp_batch_metrics[key]
        
  
    resiual_mean = torch.norm(resiual_sum, dim=1).mean()

   
    # prod_adv_average = prod_adv_average/len(ensemble_model.models)

    final_loss = final_loss
    # final_loss = final_loss / len(ensemble_model.models)

    for key in batch_metrics.keys():
        batch_metrics[key] = batch_metrics[key] / len(ensemble_model.models)
   
    alpha = 2
    final_loss += alpha*resiual_mean

    batch_metrics['logits_diff'] = logits_diff
    batch_metrics['resiual_mean'] = resiual_mean.item()
    # final_loss = final_loss/len(ensemble_model.models)

    return final_loss, batch_metrics


def trades_loss_plus_resiual_sum(ensemble_model, x_natural, y, step_size=0.002, epsilon=0.031, perturb_steps=40,  attack_type='l_infty'):
    """
    Adversarial training.
    """

    # define KL-loss
    loss_func = torch.nn.CrossEntropyLoss()
    
    final_loss = 0
    logits_adv_average = 0
    prod_adv_average = 0
    resiual_sum = 0
    batch_metrics = {}

    x_advs = []
    
    for model in ensemble_model.models:
        x_adv = PGD_attack(model, x_natural, y, epsilon, perturb_steps, step_size, attack_type, bias=None)
        x_advs.append(x_adv)


    for idx, model in enumerate(ensemble_model.models):
        model.train()
        logits_adv = model(x_advs[idx])
        loss_robust = loss_func(logits_adv, y)
        
        logits_natural = model(x_natural)
        loss_natural = loss_func(logits_natural, y)

        logits_adv_average += logits_adv
        # prod_adv_average += F.softmax(logits_adv, dim=1)

        # resiual += (logits_adv - logits_natural)
        # # final_loss += loss_natural
        final_loss += loss_robust
        # final_loss = final_loss + loss_robust

        # alpha = 0.2
        resiual = (logits_adv - logits_natural)
        # final_loss += alpha*torch.norm(resiual, dim=1).mean()

        if idx == 0:
            logits_pre = logits_adv
        else:
            logits_diff = torch.norm(logits_adv - logits_pre, dim=1).mean().item()
        
        resiual_sum += resiual
        
        if len(batch_metrics) == 0:
            batch_metrics = {'clean_loss_avg': loss_natural.item(), 'adv_loss': loss_robust.item(), 'clean_acc_num_avg': accuracy(y, logits_natural.detach())*float(y.size(0)), 'adv_acc_num_avg': accuracy(y, logits_adv.detach())*float(y.size(0))}
        else:
            tmp_batch_metrics = {'clean_loss_avg': loss_natural.item(), 'adv_loss': loss_robust.item(), 'clean_acc_num_avg': accuracy(y, logits_natural.detach())*float(y.size(0)), 'adv_acc_num_avg': accuracy(y, logits_adv.detach())*float(y.size(0))}
            for key in batch_metrics.keys():
                batch_metrics[key] = batch_metrics[key] + tmp_batch_metrics[key]
        
        # if keep_clean:
        #     final_loss = final_loss + loss_natural + loss_robust
        # else:
        #     final_loss = final_loss + loss_robust
    # breakpoint()
    resiual_mean = torch.norm(resiual_sum, dim=1).mean()
    # breakpoint()
    logits_adv_average = logits_adv_average/len(ensemble_model.models)
    # prod_adv_average = prod_adv_average/len(ensemble_model.models)

    final_loss = final_loss
    # final_loss = final_loss / len(ensemble_model.models)

    for key in batch_metrics.keys():
        batch_metrics[key] = batch_metrics[key] / len(ensemble_model.models)
   
    alpha = 0.5
    # final_loss = alpha*final_loss + (1-alpha)*loss_func(logits_adv_average, y)
    final_loss += alpha*resiual_mean

    batch_metrics['logits_diff'] = logits_diff
    batch_metrics['resiual_mean'] = resiual_mean.item()
    # final_loss = final_loss/len(ensemble_model.models)

    return final_loss, batch_metrics











def diff_init_loss(ensemble_model, datas, targets, step_size, epsilon, perturb_steps, attack_type):
    optimized_model = ensemble_model.models[-1]
    ensemble_model.models.pop()
    ensemble_model.eval()
    adv_data = PGD_attack(ensemble_model, datas, targets, epsilon, perturb_steps, step_size, attack_type, bias=None)
    per1 = (adv_data - datas)
    optimized_model.eval()
    adv_data_sub_model = PGD_attack(optimized_model, datas, targets, epsilon, perturb_steps, step_size, attack_type, bias=per1)
    optimized_model.train()

    loss_func = torch.nn.CrossEntropyLoss()

    logits_adv = optimized_model(adv_data_sub_model)
    loss_robust = loss_func(logits_adv, targets)
    logits_natural = optimized_model(datas)
    loss_natural = loss_func(logits_natural, targets)

    ensemble_model.models.append(optimized_model)

    batch_metrics = {'clean_loss': loss_natural.item(), 'adv_loss': loss_robust.item(), 'clean_acc_num': accuracy(targets, logits_natural.detach())*float(targets.size(0)), 'adversarial_acc_num': accuracy(targets, logits_adv.detach())*float(targets.size(0)), 'misalign_loss':0, 'cos_loss':0, 'norm_loss':0}

    return loss_robust, batch_metrics

    
def diff_init_loss_two_model(model1, model2, datas, targets, step_size, epsilon, perturb_steps, attack_type):

    if random.randint(0,1) == 0:
        model1, model2 = model2, model1

    model1.eval()
    adv_data1 = PGD_attack(model1, datas, targets, epsilon, perturb_steps, step_size, attack_type, bias=None)
    model1.train()
    per1 = (adv_data1 - datas)
    model2.eval()
    adv_data2 = PGD_attack(model2, datas, targets, epsilon, perturb_steps, step_size, attack_type, bias = -per1)
    model2.train()

    loss_func = torch.nn.CrossEntropyLoss()

    logits_adv_1 = model1(adv_data1)
    loss_robust1 = loss_func(logits_adv_1, targets)
    logits_natural1 = model1(datas)
    loss_natural1 = loss_func(logits_natural1, targets)

    logits_adv_2 = model2(adv_data2)
    loss_robust2 = loss_func(logits_adv_2, targets)
    logits_natural2 = model2(datas)
    loss_natural2 = loss_func(logits_natural2, targets)

    loss_robust = (loss_robust1+loss_robust2)/2
    loss_natural = (loss_natural1+loss_natural2)/2

    logits_natural = (logits_natural1+logits_natural2)/2
    logits_adv = (logits_adv_1+logits_adv_2)/2


    batch_metrics = {'clean_loss': loss_natural.item(), 'adv_loss': loss_robust.item(), 'clean_acc_num': accuracy(targets, logits_natural.detach())*float(targets.size(0)), 'adversarial_acc_num': accuracy(targets, logits_adv.detach())*float(targets.size(0)), 'misalign_loss':0, 'cos_loss':0, 'norm_loss':0}

    return loss_robust, batch_metrics



def diff_init_loss_one_model(model, datas, targets, step_size, epsilon, perturb_steps, attack_type):

    model.eval()
    adv_data1 = PGD_attack(model, datas, targets, epsilon, perturb_steps, step_size, attack_type, bias = None)

    per1 = (adv_data1 - datas)
    adv_data2 = PGD_attack(model, datas, targets, epsilon, perturb_steps, step_size, attack_type, bias = -per1)
    model.train()

    loss_func = torch.nn.CrossEntropyLoss()

    logits_adv_1 = model(adv_data1)
    loss_robust1 = loss_func(logits_adv_1, targets)
    logits_natural1 = model(datas)
    loss_natural1 = loss_func(logits_natural1, targets)

    logits_adv_2 = model(adv_data2)
    loss_robust2 = loss_func(logits_adv_2, targets)
    logits_natural2 = model(datas)
    loss_natural2 = loss_func(logits_natural2, targets)

    loss_robust = (loss_robust1+loss_robust2)/2
    loss_natural = (loss_natural1+loss_natural2)/2

    logits_natural = (logits_natural1+logits_natural2)/2
    logits_adv = (logits_adv_1+logits_adv_2)/2


    batch_metrics = {'clean_loss': loss_natural.item(), 'adv_loss': loss_robust.item(), 'clean_acc_num': accuracy(targets, logits_natural.detach())*float(targets.size(0)), 'adversarial_acc_num': accuracy(targets, logits_adv.detach())*float(targets.size(0)), 'misalign_loss':0, 'cos_loss':0, 'norm_loss':0}

    return loss_robust, batch_metrics







def joint_independent_loss(ensemble_model, x_natural, y, step_size=0.002, epsilon=0.031, perturb_steps=40,  attack='linf-pgd', keep_clean = False):
    loss1, batch_metric1 = adv_loss_naive(ensemble_model, x_natural, y, step_size, epsilon, perturb_steps,  attack, keep_clean)
    loss2, batch_metric2 = adv_loss_naive_multi(ensemble_model, x_natural, y, step_size, epsilon, perturb_steps,  attack, keep_clean)
    return (loss1+loss2)/2, batch_metric1


    
    