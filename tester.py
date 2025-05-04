from utils import progress_bar
from attacks.pgd_non_label_leaking import LinfPGDAttack, L2PGDAttack
from attacks.apgd import LinfAPGDAttack
from attacks.pgd import PGD_attack, PGD_attack_two_class_classification, PGD_attack_regression, FGSM_attack, PGD_attack_pro, PGD_attack_EOT
from attacks.mora import MORA
from attacks.deepfool import DeepfoolLinfAttack
from autoattack import AutoAttack
from autoattack.square import SquareAttack
import torch 
import os
import torch.nn.functional as F
import torchvision.transforms as transforms


def test_and_save_clean(args, net, testloader, device, epoch, best_acc, save=False): 
    net.eval()

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    with open('./output/result_acc_%s_%s.txt'%(args['model'], args['method']), 'a') as f:
        f.write('Epoch: %d, Batch: %d '%(epoch, batch_idx))
        f.write('Adv_Loss: %.3f| Adv_Acc: %.3f%% (%d)\n'% (test_loss/(batch_idx+1), 
                            100.*correct/total, 
                            total))

    # Save checkpoint.
    if save:
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc

    return best_acc


def test_and_save_adv(args, net, testloader, device, epoch, best_adv_acc, save=True, name = 'ckpt', attack_method = 'PGD_pro', surrogate_model = None): 
    net.eval()
    
    
    test_adv_loss_1 = 0
    test_adv_correct_1 = 0
    total_num = 0

    loss_fn = torch.nn.CrossEntropyLoss()

    if attack_method == 'AA':
        aa = AutoAttack(net, norm = "Linf", eps = args['perturbation_size'], version = 'standard', device = device, verbose= False)
    elif attack_method == 'APGD':
        aa = LinfAPGDAttack(device, net, eps=args['perturbation_size'])
    elif attack_method == 'MORA':
        norm = 'Linf' if args['attack_type'] == 'l_infty' else 'L2'
        MoraAttack = MORA(net, device = device, norm = norm, eps = args['perturbation_size'], n_iter = args['perturb_steps'],  ensemble_pattern = 'logits')
        MoraAttack.attacks_to_run = ['mora-ce']
    elif attack_method == 'DeepFool':
        DFAttack = DeepfoolLinfAttack(net, nb_iter = args['perturb_steps'], eps=args['perturbation_size'])
    elif attack_method == 'BlackBox':
        BBAttack = SquareAttack(net, p_init=.8, n_queries=5000, eps=args['perturbation_size'], norm='Linf',
                n_restarts=1, seed=42, verbose=False, device=device, resc_schedule=False)


    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        if attack_method == 'PGD':
            adv_inputs_1 = PGD_attack(net, inputs, targets, args['perturbation_size'], args['perturb_steps'], args['step_size'], type=args['attack_type'], bias=None)

        elif attack_method == 'PGD_EOT':
            adv_inputs_1 = PGD_attack_EOT(net, inputs, targets, args['perturbation_size'], args['perturb_steps'], args['step_size'], type=args['attack_type'], bias=None)
        elif attack_method == 'PGD_pro':    
            adv_inputs_1 = PGD_attack_pro(net, inputs, targets, args['perturbation_size'], args['perturb_steps'], args['step_size'], type=args['attack_type'], bias=None)
        elif attack_method == 'FGSM':
            adv_inputs_1 = FGSM_attack(net, inputs, targets, args['perturbation_size'], type=args['attack_type'])
        elif attack_method == 'AA':
            adv_inputs_1 = aa.run_standard_evaluation(inputs, targets, bs=256)
        elif attack_method == 'MORA':
            adv_inputs_1 = MoraAttack.run_standard_evaluation(inputs, targets)
            tmp = adv_inputs_1 - inputs
            if tmp.max()>0.315 or tmp.min()<-0.315 or adv_inputs_1.max() > 1 or adv_inputs_1.min() < -1:
                print('Error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                breakpoint()
        elif attack_method == 'DeepFool':
            adv_inputs_1 = DFAttack.perturb(inputs, targets)
            tmp = adv_inputs_1 - inputs
            if tmp.max()>0.315 or tmp.min()<-0.315 or adv_inputs_1.max() > 1 or adv_inputs_1.min() < -1:
                print('Error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                breakpoint()
        elif attack_method == 'APGD':
            adv_inputs_1, _ = aa.perturb(inputs, targets)
        elif attack_method == 'BlackBox':
            adv_inputs_1 = BBAttack.perturb(inputs, targets)
        elif attack_method == 'PGD_surrogate':
            adv_inputs_1 = PGD_attack(surrogate_model, inputs, targets, args['perturbation_size'], args['perturb_steps'], args['step_size'], type=args['attack_type'], bias=None)
        
        transform = transforms.RandomRotation(degrees=(-30, 30))
        with torch.no_grad():
            if attack_method == 'PGD_EOT':
                for _ in range(20):
                    rotated_image = transform(adv_inputs_1)
                    adv_outputs_1 = net(rotated_image)
                    loss1 = loss_fn(adv_outputs_1, targets)
                    test_adv_loss_1 += loss1.item()
                    _, predicted1 = adv_outputs_1.max(1)

                    total_num += targets.size()[0]
                    test_adv_correct_1 += predicted1.eq(targets).sum().item()
            else:
                if attack_method == 'MORA':
                    adv_outputs_1 = net(adv_inputs_1)[-1]
                else: 
                    adv_outputs_1 = net(adv_inputs_1)

                loss1 = loss_fn(adv_outputs_1, targets)
                test_adv_loss_1 += loss1.item()
                _, predicted1 = adv_outputs_1.max(1)

                total_num += targets.size()[0]
                test_adv_correct_1 += predicted1.eq(targets).sum().item()

        progress_bar(batch_idx, len(testloader), 'Adv_Loss: %.3f| Adv_Acc: %.3f%% (%d)'
                        % (test_adv_loss_1/(batch_idx+1),
                            100.*test_adv_correct_1/total_num,  
                            total_num))

    if save:
        with open('./output/result_adv_%s_%s.txt'%(args['model'], args['method']), 'a') as f:
            f.write('Epoch: %d, Batch: %d '%(epoch, batch_idx))
            f.write('Adv_Loss: %.3f| Adv_Acc: %.3f %% (%d)\n'% (test_adv_loss_1/(batch_idx+1), 
                                100.*test_adv_correct_1/total_num,  
                                total_num))
            
    adv_acc = 100.*test_adv_correct_1/total_num
    # Save checkpoint.
    if adv_acc > best_adv_acc:
        best_adv_acc = adv_acc
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir('checkpoint/%s_%s_models'%(args['model'], args['method'])):
            os.mkdir('checkpoint/%s_%s_models'%(args['model'], args['method']))
        if save:
            print('Saving..')
            if args['method'][0:3] == 'PDD':
                for idx, (model, fc) in enumerate(zip(net.extractors, net.fc)):
                    state = {
                        'net': model.state_dict(),
                        'adv_acc': adv_acc,
                        'epoch': epoch,
                    }
                    torch.save(state, './checkpoint/%s_%s_models/%s_model_%d.pth'%(args['model'], args['method'], name, idx))
                    state = {
                        'net': fc.state_dict(),
                        'adv_acc': adv_acc,
                        'epoch': epoch,
                    }
                    torch.save(state, './checkpoint/%s_%s_models/%s_fc_%d.pth'%(args['model'], args['method'], name, idx))
            else:
                if hasattr(net, 'models'):
                    for model_idx, model in enumerate(net.models):
                        state = {
                            'net': model.state_dict(),
                            'adv_acc': adv_acc,
                            'epoch': epoch,
                        }
                        torch.save(state, './checkpoint/%s_%s_models/%s_%d.pth'%(args['model'], args['method'], name, model_idx))
                else:
                    state = {
                        'net': net.state_dict(),
                        'adv_acc': adv_acc,
                        'epoch': epoch,
                    }
                    torch.save(state, './checkpoint/%s_%s_models/%s.pth'%(args['model'], args['method'], name))
        
    return best_adv_acc


def cal_var(args, ensemble_model, testloader, device, epoch, best_adv_acc, save=True):
    
    pred_vars = []
    num_samples = 0

    cos_res = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        num_samples += inputs.shape[0]
        preds = []
        pred_advs = []
        for model in ensemble_model.models:
            inputs = inputs.to(device)
            targets = targets.to(device)
            adv_inputs = PGD_attack(model, inputs, targets, args['perturbation_size'], args['perturb_steps'], args['step_size'], type=args['attack_type'], bias=None)
            pred = model(inputs)
            pred_adv = model(adv_inputs)

            pred = F.softmax(pred, dim = 1)
            pred_adv = F.softmax(pred_adv, dim = 1)

            preds.append(pred.detach())
            pred_advs.append(pred_adv.detach())

        preds = torch.stack(preds)
        pred_advs = torch.stack(pred_advs)
        preds_mean = torch.mean(preds, dim = 0)

        preds_diff = pred_advs - preds_mean
        
        for i in range(preds_diff.shape[0]):
            for j in range(i, preds_diff.shape[0]):
                cos_res += F.cosine_similarity(preds_diff[i], preds_diff[j]).sum()


        # for i in range(preds.shape[0]):
        #     pred_advs[i] - preds_mean
            # preds[i] = preds[i] - preds_mean
        
        # pred_var = torch.norm(preds, dim = 2)
        # pred_var = pred_var.sum()/preds.shape[2]
        # pred_vars.append(pred_var.item())

    cos_res = cos_res/(preds_diff.shape[0]*(preds_diff.shape[0]-1))/num_samples

    # return sum(pred_vars)/num_samples
    return cos_res.detach().item()

def accuracy_regression(true, preds):
    """
    Computes multi-class accuracy.
    Arguments:
        true (torch.Tensor): true labels.
        preds (torch.Tensor): predicted labels.
    Returns:
        Multi-class accuracy.
    """
    accuracy = ((true*preds.flatten())>0).sum().float()/float(true.size(0))

    return accuracy.item()

def test_and_save_clean_two_class_regression(args, net, testloader, device, epoch, best_acc, save=False, name = 'ckpt'): 
    net.eval()

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(outputs.flatten(), targets)

            test_loss += loss.item()
            total += targets.size(0)
            correct += accuracy_regression(targets, outputs.flatten())*targets.size(0)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    with open('./output/result_acc.txt', 'a') as f:
        f.write('Epoch: %d, Batch: %d '%(epoch, batch_idx))
        f.write('Adv_Loss: %.3f| Adv_Acc: %.3f%% (%d)\n'% (test_loss/(batch_idx+1), 
                            100.*correct/total, 
                            total))

    # Save checkpoint.
    if save:
        acc = 100.*correct/total
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir('checkpoint/%s_models'%args['method']):
            os.mkdir('checkpoint/%s_models'%args['method'])
        if acc > best_acc:
            print('Saving..')
            if hasattr(net, 'models'):
                for model_idx, model in enumerate(net.models):
                    state = {
                        'net': model.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                    }
                    torch.save(state, './checkpoint/%s_models/%s_%d.pth'%(args['method'], name, model_idx))
            else:
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                torch.save(state, './checkpoint/%s.pth'%name)

    return best_acc

def test_and_save_adv_two_class_regression(args, net, testloader, device, epoch, best_adv_acc, save=True, name = 'ckpt', attack_method = 'PGD'): 

    net.eval()

    test_adv_loss_1 = 0
    test_adv_correct_1 = 0
    total_num = 0

    loss_fn = torch.nn.MSELoss()


    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        if attack_method == 'PGD':
            adv_inputs_1 = PGD_attack_regression(net, inputs, targets, args['perturbation_size'], args['perturb_steps'], args['step_size'], type=args['attack_type'], bias=None)
        elif attack_method == 'FGSM':
            pass
        elif attack_method == 'AA':
            pass

        with torch.no_grad():
            adv_outputs_1 = net(adv_inputs_1)
            
            loss1 = loss_fn(adv_outputs_1.flatten(), targets)
            test_adv_loss_1 += loss1.item()*targets.shape[0]

            total_num += targets.size()[0]
            test_adv_correct_1 += accuracy_regression(targets, adv_outputs_1)*targets.size()[0]


        progress_bar(batch_idx, len(testloader), 'Adv_Loss: %.3f| Adv_Acc: %.3f%% (%d)'
                        % (test_adv_loss_1/(total_num),
                            100.*test_adv_correct_1/total_num,  
                            total_num))


    with open('./output/result_adv.txt', 'a') as f:
        f.write('Epoch: %d, Batch: %d '%(epoch, batch_idx))
        f.write('Adv_Loss: %.3f| Adv_Acc: %.3f %% (%d)\n'% (test_adv_loss_1/(batch_idx+1), 
                            100.*test_adv_correct_1/total_num,  
                            total_num))
            
    adv_acc = 100.*test_adv_correct_1/total_num
    # Save checkpoint.
    if adv_acc > best_adv_acc:
        best_adv_acc = adv_acc
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir('checkpoint/%s_%s_models'%(args['model'], args['method'])):
            os.mkdir('checkpoint/%s_%s_models'%(args['model'], args['method']))
        if save:
            print('Saving..')
            if hasattr(net, 'models'):
                for model_idx, model in enumerate(net.models):
                    state = {
                        'net': model.state_dict(),
                        'adv_acc': adv_acc,
                        'epoch': epoch,
                    }
                    torch.save(state, './checkpoint/%s_%s_models/%s_%d.pth'%(args['model'], args['method'], name, model_idx))
            else:
                state = {
                    'net': net.state_dict(),
                    'adv_acc': adv_acc,
                    'epoch': epoch,
                }
                torch.save(state, './checkpoint/%s_%s_models/%s.pth'%(args['model'], args['method'], name))
        
    return best_adv_acc, test_adv_loss_1/total_num



def two_class_cross_entropy(logit_y, targets):
    pos_y = 1/(1+torch.exp(-logit_y))
    neg_y = 1 - pos_y

    cro_entro = -((pos_y.flatten()*torch.log(targets+1e-4)).sum() + (neg_y.flatten()*torch.log(1-targets+1e-4)).sum())

    # breakpoint()
    return cro_entro/targets.shape[0]


def test_and_save_clean_two_class_classification(net, testloader, device, epoch, best_acc, save=False): 
    net.eval()

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss_fn = two_class_cross_entropy
            loss = loss_fn(outputs, targets)

            test_loss += loss.item()
            total += targets.size(0)
            correct += accuracy_regression(2*targets-1, outputs.flatten())*targets.size(0)


            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    with open('./output/result_acc.txt', 'a') as f:
        f.write('Epoch: %d, Batch: %d '%(epoch, batch_idx))
        f.write('Adv_Loss: %.3f| Adv_Acc: %.3f%% (%d)\n'% (test_loss/(batch_idx+1), 
                            100.*correct/total, 
                            total))

    # Save checkpoint.
    if save:
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc

    return best_acc


def test_and_save_adv_two_class_classification(args, net, testloader, device, epoch, best_adv_acc, save=True, name = 'ckpt', attack_method = 'PGD'): 

    net.eval()

    test_adv_loss_1 = 0
    test_adv_correct_1 = 0
    total_num = 0

    loss_fn = two_class_cross_entropy

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # breakpoint()
        inputs = inputs.to(device)
        targets = targets.to(device)

        if attack_method == 'PGD':
            adv_inputs_1 = PGD_attack_two_class_classification(net, inputs, targets, args['perturbation_size'], args['perturb_steps'], args['step_size'], type=args['attack_type'], bias=None)
        elif attack_method == 'FGSM':
            pass
        elif attack_method == 'AA':
            pass

        with torch.no_grad():
            adv_outputs_1 = net(adv_inputs_1)
            
            loss1 = loss_fn(adv_outputs_1, targets)
            test_adv_loss_1 += loss1.item()*targets.shape[0]

            total_num += targets.size()[0]
            test_adv_correct_1 += accuracy_regression(2*targets-1, adv_outputs_1.flatten())*targets.size(0)


        progress_bar(batch_idx, len(testloader), 'Adv_Loss: %.3f| Adv_Acc: %.3f%% (%d)'
                        % (test_adv_loss_1/(total_num),
                            100.*test_adv_correct_1/total_num,  
                            total_num))


    with open('./output/result_adv.txt', 'a') as f:
        f.write('Epoch: %d, Batch: %d '%(epoch, batch_idx))
        f.write('Adv_Loss: %.3f| Adv_Acc: %.3f %% (%d)\n'% (test_adv_loss_1/(batch_idx+1), 
                            100.*test_adv_correct_1/total_num,  
                            total_num))
            
    adv_acc = 100.*test_adv_correct_1/total_num
    # Save checkpoint.
    if adv_acc > best_adv_acc:
        best_adv_acc = adv_acc
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir('checkpoint/%s_%s_models'%(args['model'], args['method'])):
            os.mkdir('checkpoint/%s_%s_models'%(args['model'], args['method']))
        if save:
            print('Saving..')
            if hasattr(net, 'models'):
                for model_idx, model in enumerate(net.models):
                    state = {
                        'net': model.state_dict(),
                        'adv_acc': adv_acc,
                        'epoch': epoch,
                    }
                    torch.save(state, './checkpoint/%s_%s_models/%s_%d.pth'%(args['model'], args['method'], name, model_idx))
            else:
                state = {
                    'net': net.state_dict(),
                    'adv_acc': adv_acc,
                    'epoch': epoch,
                }
                torch.save(state, './checkpoint/%s_%s_models/%s.pth'%(args['model'], args['method'], name))
        
    return best_adv_acc, test_adv_loss_1/total_num




