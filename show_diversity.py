import torch
import random
import numpy as np

from data.cifar10 import load_cifar10_two_class_regression, load_cifar10_two_class_classification
from torch.utils.data import DataLoader
from models.CNN import CNN_NET_regression, CNN_NET_classification
from models.ensemble import EnsembleModel
from tester import test_and_save_adv_two_class_regression, test_and_save_adv_two_class_classification

def calcul_gradient_wrt_model(model, x, y, if_optimized):
    if not if_optimized:
        model.eval()
    x.requires_grad = True
    outputs = model(x)
    grad = torch.autograd.grad(outputs.mean(), x, retain_graph=if_optimized, create_graph=if_optimized)[0]
    if not if_optimized:
        model.train()
    return grad * x.size()[0]

def cal_cos_value_wrt_model(model1, model2, test_loader, device):
    import torch.nn.functional as F
    cos_value_sum = 0
    num_sample = 0
    for x,y in test_loader:
        x,y = x.to(device), y.to(device)
        grad1 = calcul_gradient_wrt_model(model1, x, y, True)
        grad1 = grad1.view([grad1.size()[0], -1])
        grad2 = calcul_gradient_wrt_model(model2, x, y, True)
        grad2 = grad2.view([grad2.size()[0], -1])
        
        num_sample += y.shape[0]

        cos_value_sum += F.cosine_similarity(grad1, grad2).sum()
    
    return cos_value_sum.detach().item()/num_sample

def cal_cos_value_wrt_model_multi_models(model_list, test_loader, device):
    cos_value_sum = 0
    model_num = len(model_list)
    for i in range(model_num):
        for j in range(i+1, model_num):
            cos_value = cal_cos_value_wrt_model(model_list[i], model_list[j], test_loader, device)
            cos_value_sum += cos_value
    return cos_value_sum/((model_num)*(model_num-1)/2)


def cal_div_multi_models(model_list, test_loader, epsilon, device, attack_type):
    import torch.nn.functional as F
    model_num = len(model_list)
    div1 = 0
    div2 = 0
    div3 = 0
    num_sample = 0
    for x,y in test_loader:
        x,y = x.to(device), y.to(device)
        grad_list = []
        pred_list = []
        grad_sum = 0
        pred_sum = 0
        for model in model_list:
            tmp_grad = calcul_gradient_wrt_model(model, x, y, True)
            tmp_grad = tmp_grad.view([tmp_grad.size()[0], -1])
            grad_list.append(tmp_grad)
            grad_sum = grad_sum + tmp_grad
            tmp_pred = model(x).flatten().detach()
            pred_list.append(tmp_pred)
            pred_sum = pred_sum + tmp_pred
        grad_avg = grad_sum/model_num
        pred_avg = pred_sum/model_num
        num_sample += y.shape[0]
        for pred in pred_list:
            div1 += ((pred - pred_avg)**2).sum()
        if attack_type == 'l_2':
            for grad in grad_list:
                # div1 += (torch.norm(grad-grad_avg, p=1, dim=1)**2).sum()
                div2 += (torch.norm(grad, p=2, dim=1)**2 - torch.norm(grad_avg, p=2, dim=1)**2).sum()
                
            for grad, pred in zip(grad_list, pred_list):
                diff = torch.abs(pred - y)
                diff_avg = torch.abs(pred_avg - y)
                div3 += (torch.norm(grad, p=2, dim=1)*diff - torch.norm(grad_avg, p=2, dim=1)*diff_avg).sum()
        else: 
            for grad in grad_list:
                # div1 += (torch.norm(grad-grad_avg, p=1, dim=1)**2).sum()
                div2 += (torch.norm(grad, p=1, dim=1)**2 - torch.norm(grad_avg, p=1, dim=1)**2).sum()
                
            for grad, pred in zip(grad_list, pred_list):
                diff = torch.abs(pred - y)
                diff_avg = torch.abs(pred_avg - y)
                div3 += (torch.norm(grad, p=1, dim=1)*diff - torch.norm(grad_avg, p=1, dim=1)*diff_avg).sum()
    div1 = div1/model_num/num_sample
    div2 = div2*(epsilon**2)/model_num/num_sample
    div3 = div3*(epsilon*2)/model_num/num_sample
    
    return div1.detach().item(), div2.detach().item(), div3.detach().item()

def test_and_show_the_diversity_two_class_regression(attackargs, trainargs, device):

    train_dataset_sup, test_dataset = load_cifar10_two_class_regression(5, 6, './data/data')

    train_loader = DataLoader(train_dataset_sup,
                              batch_size=trainargs['batch_size'],
                              shuffle=True, num_workers=1)

    test_loader = DataLoader(test_dataset,
                              batch_size=trainargs['batch_size'],
                              shuffle=True, num_workers=1)

    adv_loss_avg_list = []
    adv_loss_ens_list = []
    adv_acc_avg_list = []
    adv_acc_ens_list = []
    cos_list = []
    div1_list = []
    div2_list = []
    div3_list = []


    for model_num in range(1, 13):

        model_list = []
        for _ in range(model_num):
            model_list.append(CNN_NET_regression().to(device))

        best_acc = 0
        best_adv_acc = 0

        adv_acc_avgs = []
        adv_acc_enss = []

        adv_loss_avgs = []
        adv_loss_enss = []

        cos_val_models = []
        div_1s = []
        div_2s = []
        div_3s = []



        for _ in range(5):

            adv_acc_avg = 0
            adv_loss_avg = 0

            for idx, model_idx in enumerate(random.sample([i for i in range(10)], model_num)):
                print(model_idx)
                saved = torch.load('./checkpoint/CNN_NET_regression_Two_class_train_regression_40_models/'+str(model_idx)+'_last_ckpt.pth')
                model_dict = saved['net']
                model_list[idx].load_state_dict(model_dict)
                adv_acc, adv_loss = test_and_save_adv_two_class_regression(attackargs, model_list[idx], test_loader, device, 0, 0, save= False)
                adv_acc_avg += adv_acc
                adv_loss_avg += adv_loss

            adv_acc_avg = adv_acc_avg/model_num
            adv_loss_avg = adv_loss_avg/model_num


            ensemble_model = EnsembleModel(model_list)
            adv_acc_ens, adv_loss_ens = test_and_save_adv_two_class_regression(attackargs, ensemble_model, test_loader, device, 0, 0, save = False)

            adv_acc_avgs.append(adv_acc_avg)
            adv_loss_avgs.append(adv_loss_avg)

            adv_acc_enss.append(adv_acc_ens)
            adv_loss_enss.append(adv_loss_ens)
            
            if model_num == 1:
                cos_val_model = 0
            else:
                cos_val_model = cal_cos_value_wrt_model_multi_models(model_list, test_loader, device)
                # cos_val_model = cal_cos_value_wrt_loss_multi_models(model_list, test_loader, device)
            cos_val_models.append(cos_val_model)

            if model_num == 1:
                div_1, div_2, div_3 = 0, 0, 0
            else:
                div_1, div_2, div_3 = cal_div_multi_models(model_list, test_loader, attackargs['perturbation_size'], device, attackargs['attack_type'])

            div_1s.append(div_1)
            div_2s.append(div_2)
            div_3s.append(div_3)
        
        with open('./diversity.txt', 'a') as f:
            f.write('Model number:%d'%model_num)
            f.write('\n')
            f.write('adv_acc_avg:%.5f+-%.5f'%(np.mean(adv_acc_avgs), np.std(adv_acc_avgs, ddof=1)))
            f.write('\n')
            f.write('adv_acc_ens:%.5f+-%.5f'%(np.mean(adv_acc_enss), np.std(adv_acc_enss, ddof=1)))
            f.write('\n')
            f.write('adv_loss_avg:%.5f+-%.5f'%(np.mean(adv_loss_avgs), np.std(adv_loss_avgs, ddof=1)))
            f.write('\n')
            f.write('adv_loss_ens:%.5f+-%.5f'%(np.mean(adv_loss_enss), np.std(adv_loss_enss, ddof=1)))
            f.write('\n')
            f.write('div_1:%.5f+-%.5f'%(np.mean(div_1s), np.std(div_1s, ddof=1)))
            f.write('\n')
            f.write('div_2:%.5f+-%.5f'%(np.mean(div_2s), np.std(div_2s, ddof=1)))
            f.write('\n')
            f.write('div_3:%.5f+-%.5f'%(np.mean(div_3s), np.std(div_3s, ddof=1)))
            f.write('\n')
            f.write('cos_val_model:%.5f+-%.5f'%(np.mean(cos_val_models), np.std(cos_val_models, ddof=1)))
            f.write('\n')
        
        adv_acc_avg_list.append((np.mean(adv_acc_avgs), np.std(adv_acc_avgs, ddof=1)))
        adv_acc_ens_list.append((np.mean(adv_acc_enss), np.std(adv_acc_enss, ddof=1)))
        adv_loss_avg_list.append((np.mean(adv_loss_avgs), np.std(adv_loss_avgs, ddof=1)))
        adv_loss_ens_list.append((np.mean(adv_loss_enss), np.std(adv_loss_enss, ddof=1)))
        cos_list.append((np.mean(cos_val_models), np.std(cos_val_models, ddof=1)))
        div1_list.append((np.mean(div_1s), np.std(div_1s, ddof=1)))
        div2_list.append((np.mean(div_2s), np.std(div_2s, ddof=1)))
        div3_list.append((np.mean(div_3s), np.std(div_3s, ddof=1)))
    
    print(adv_acc_avg_list)
    print(adv_acc_ens_list)
    print(adv_loss_avg_list)
    print(adv_loss_ens_list)
    print(cos_list)
    print(div1_list)
    print(div2_list)
    print(div3_list)

    with open('./diversity.txt', 'a') as f:
        f.write('adv_acc_avg_list: ' + str(adv_acc_avg_list))
        f.write('\n')
        f.write('adv_acc_ens_list: ' + str(adv_acc_ens_list))
        f.write('\n')
        f.write('adv_loss_avg_list: ' + str(adv_loss_avg_list))
        f.write('\n')
        f.write('adv_loss_ens_list: ' + str(adv_loss_ens_list))
        f.write('\n')
        f.write('cos_list: ' + str(cos_list))
        f.write('\n')
        f.write('div1_list: ' + str(div1_list))
        f.write('\n')
        f.write('div2_list: ' + str(div2_list))
        f.write('\n')
        f.write('div3_list: ' + str(div3_list))
        f.write('\n')        
    return 

def cal_div_multi_models_classification(model_list, test_loader, epsilon, device, attack_type):
    import torch.nn.functional as F
    if attack_type == 'l_infty':
        P = 1
    else: 
        P = 2 
    model_num = len(model_list)
    div1 = 0
    div2 = 0
    num_sample = 0
    for x,y in test_loader:
        x,y = x.to(device), y.to(device)
        grad_list = []
        logit_list = []
        for model in model_list:
            tmp_grad = calcul_gradient_wrt_model(model, x, y, True)
            tmp_grad = tmp_grad.view([tmp_grad.size()[0], -1])
            grad_list.append(tmp_grad)
            logit = model(x)
            logit_list.append(logit)
        grad_sum = 0
        logit_sum = 0
        adv_pred_pos_list = []
        for logit, grad in zip(logit_list, grad_list):
            grad_sum += grad
            logit_sum += logit
            adv_logit_pos = logit.flatten() - (2*y-1)*torch.norm(grad, p=P, dim=1)*epsilon
            adv_pred_pos_list.append(1/(1+torch.exp(-adv_logit_pos)))
        grad_avg = grad_sum/model_num
        logit_avg = logit_sum/model_num
        adv_logit_pos_avg = logit_avg.flatten() - (2*y-1)*torch.norm(grad_avg, p=P, dim=1)*epsilon
        adv_pred_pos_avg = 1/(1+torch.exp(-adv_logit_pos_avg))
        num_sample += y.shape[0]
        kl = 0
        for adv_pred_pos in adv_pred_pos_list:
            kl += adv_pred_pos_avg*torch.log((adv_pred_pos_avg+1e-10)/(adv_pred_pos+1e-10))
            kl += (1-adv_pred_pos_avg)*torch.log((1-adv_pred_pos_avg+1e-10)/(1-adv_pred_pos+1e-10))
        div1 += kl.sum()

        R = (y-adv_pred_pos_avg)*(2*y-1)
        tmp_div2 = 0
        for grad in grad_list:
            tmp_div2 += (torch.norm(grad, p=P, dim=1) - torch.norm(grad_avg, p=P, dim=1))
        div2 += (R*tmp_div2).sum()

    div1 = div1/model_num/num_sample
    div2 = div2*epsilon/model_num/num_sample
    
    return div1.detach().item(), div2.detach().item()


def test_and_show_the_diversity_two_class_classification(attackargs, trainargs, device):

    train_dataset_sup, test_dataset = load_cifar10_two_class_classification(5, 6, './data/data')

    test_loader = DataLoader(test_dataset,
                              batch_size=trainargs['batch_size'],
                              shuffle=True, num_workers=1)


    adv_loss_avg_list = []
    adv_loss_ens_list = []
    adv_acc_avg_list = []
    adv_acc_ens_list = []
    cos_list = []
    div1_list = []
    div2_list = []


    for model_num in range(1, 13):

        model_list = []
        for _ in range(model_num):
            model_list.append(CNN_NET_classification().to(device))

        adv_acc_avgs = []
        adv_acc_enss = []

        adv_loss_avgs = []
        adv_loss_enss = []

        cos_val_models = []
        div_1s = []
        div_2s = []

        for _ in range(5):

            adv_acc_avg = 0
            adv_loss_avg = 0

            for idx, model_idx in enumerate(random.sample([i for i in range(20)], model_num)):
                print(model_idx)
                saved = torch.load('./checkpoint/CNN_NET_classification_Two_class_train_classification_60_models/'+str(model_idx)+'_last_ckpt.pth')
                model_dict = saved['net']
                model_list[idx].load_state_dict(model_dict)
                adv_acc, adv_loss = test_and_save_adv_two_class_classification(attackargs, model_list[idx], test_loader, device, 0, 0, save= False)
                adv_acc_avg += adv_acc
                adv_loss_avg += adv_loss

            adv_acc_avg = adv_acc_avg/model_num
            adv_loss_avg = adv_loss_avg/model_num


            ensemble_model = EnsembleModel(model_list)
            adv_acc_ens, adv_loss_ens = test_and_save_adv_two_class_classification(attackargs, ensemble_model, test_loader, device, 0, 0, save = False)

            adv_acc_avgs.append(adv_acc_avg)
            adv_loss_avgs.append(adv_loss_avg)

            adv_acc_enss.append(adv_acc_ens)
            adv_loss_enss.append(adv_loss_ens)

            if model_num == 1:
                div_1, div_2 = 0, 0
            else:
                div_1, div_2 = cal_div_multi_models_classification(model_list, test_loader, attackargs['perturbation_size'], device, attackargs['attack_type'])
            


            div_1s.append(div_1)
            div_2s.append(div_2)


            
            if model_num == 1:
                cos_val_model = 0
            else:
                cos_val_model = cal_cos_value_wrt_model_multi_models(model_list, test_loader, device)
                # cos_val_model = cal_cos_value_wrt_loss_multi_models(model_list, test_loader, device)

            cos_val_models.append(cos_val_model)

            
        
        with open('./diversity.txt', 'a') as f:
            f.write('Model number:%d'%model_num)
            f.write('\n')
            f.write('adv_acc_avg:%.5f+-%.5f'%(np.mean(adv_acc_avgs), np.std(adv_acc_avgs, ddof=1)))
            f.write('\n')
            f.write('adv_acc_ens:%.5f+-%.5f'%(np.mean(adv_acc_enss), np.std(adv_acc_enss, ddof=1)))
            f.write('\n')
            f.write('adv_loss_avg:%.5f+-%.5f'%(np.mean(adv_loss_avgs), np.std(adv_loss_avgs, ddof=1)))
            f.write('\n')
            f.write('adv_loss_ens:%.5f+-%.5f'%(np.mean(adv_loss_enss), np.std(adv_loss_enss, ddof=1)))
            f.write('\n')
            f.write('div_1:%.5f+-%.5f'%(np.mean(div_1s), np.std(div_1s, ddof=1)))
            f.write('\n')
            f.write('div_2:%.5f+-%.5f'%(np.mean(div_2s), np.std(div_2s, ddof=1)))
            f.write('\n')
            f.write('cos_val_model:%.5f+-%.5f'%(np.mean(cos_val_models), np.std(cos_val_models, ddof=1)))
            f.write('\n')
        
        adv_acc_avg_list.append((np.mean(adv_acc_avgs), np.std(adv_acc_avgs, ddof=1)))
        adv_acc_ens_list.append((np.mean(adv_acc_enss), np.std(adv_acc_enss, ddof=1)))
        adv_loss_avg_list.append((np.mean(adv_loss_avgs), np.std(adv_loss_avgs, ddof=1)))
        adv_loss_ens_list.append((np.mean(adv_loss_enss), np.std(adv_loss_enss, ddof=1)))
        cos_list.append((np.mean(cos_val_models), np.std(cos_val_models, ddof=1)))
        div1_list.append((np.mean(div_1s), np.std(div_1s, ddof=1)))
        div2_list.append((np.mean(div_2s), np.std(div_2s, ddof=1)))
        
    print(adv_acc_avg_list)
    print(adv_acc_ens_list)
    print(adv_loss_avg_list)
    print(adv_loss_ens_list)
    print(cos_list)
    print(div1_list)
    print(div2_list)

    with open('./diversity.txt', 'a') as f:
        f.write('adv_acc_avg_list: ' + str(adv_acc_avg_list))
        f.write('\n')
        f.write('adv_acc_ens_list: ' + str(adv_acc_ens_list))
        f.write('\n')
        f.write('adv_loss_avg_list: ' + str(adv_loss_avg_list))
        f.write('\n')
        f.write('adv_loss_ens_list: ' + str(adv_loss_ens_list))
        f.write('\n')
        f.write('cos_list: ' + str(cos_list))
        f.write('\n')
        f.write('div1_list: ' + str(div1_list))
        f.write('\n')
        f.write('div2_list: ' + str(div2_list))
        f.write('\n')
    return 

def calcul_gradient_wrt_loss(model, x, y, if_optimized):
    if not if_optimized:
        model.eval()
    x.requires_grad = True
    loss_func = torch.nn.MSELoss()
    outputs = model(x)
    loss = loss_func(outputs.flatten(), y)
    grad = torch.autograd.grad(loss, x, retain_graph=if_optimized, create_graph=if_optimized)[0]
    if not if_optimized:
        model.train()
    return grad * x.size()[0]

def cal_cos_value_wrt_loss(model1, model2, test_loader, device):
    import torch.nn.functional as F
    cos_value_sum = 0
    num_sample = 0
    for x,y in test_loader:
        x,y = x.to(device), y.to(device)
        grad1 = calcul_gradient_wrt_loss(model1, x, y, True)
        grad1 = grad1.view([grad1.size()[0], -1])
        grad2 = calcul_gradient_wrt_loss(model2, x, y, True)
        grad2 = grad2.view([grad2.size()[0], -1])
        num_sample += y.shape[0]
        cos_value_sum += F.cosine_similarity(grad1, grad2).sum()
    return cos_value_sum.detach().item()/num_sample

def cal_cos_value_wrt_loss_multi_models(model_list, test_loader, device):
    cos_value_sum = 0
    model_num = len(model_list)
    for i in range(model_num):
        for j in range(i+1, model_num):
            cos_value = cal_cos_value_wrt_loss(model_list[i], model_list[j], test_loader, device)
            cos_value_sum += cos_value
    return cos_value_sum/((model_num)*(model_num-1)/2)