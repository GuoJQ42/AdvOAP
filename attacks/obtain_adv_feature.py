from utils import progress_bar
from attacks.pgd_non_label_leaking import LinfPGDAttack, L2PGDAttack
# from attacks.apgd import LinfAPGDAttack
from attacks.pgd import PGD_attack, PGD_ensemble_attack, PGD_attack_regression, FGSM_attack
import torch 
import os
import torch.nn.functional as F
from loss.diversity_loss import diversity_loss_multi_regression
import numpy as np


def test_and_save_adv_features(args, net, testloader, device, attack_method, path): 
    net.eval()

    # loss_fn = torch.nn.CrossEntropyLoss()
    # AutoAttack = LinfAPGDAttack(net, eps=8/255)
    # AutoAttack = L2PGDAttack(net, eps=128/255)

    final_features = None
    final_targets = None

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        if attack_method == 'PGD':
            adv_inputs_1 = PGD_attack(net, inputs, targets, args['perturbation_size'], args['perturb_steps'], args['step_size'], type=args['attack_type'], bias=None)
        elif attack_method == 'FGSM':
            adv_inputs_1 = FGSM_attack(net, inputs, targets, args['perturbation_size'], type=args['attack_type'])
        elif attack_method == 'AA':
            pass
        
        with torch.no_grad():
            final_feature = net.get_final_feature(adv_inputs_1).cpu().numpy()
        
        if final_features is None:
            final_features = final_feature
        else:
            final_features = np.vstack([final_features, final_feature]) 
        
        if final_targets is None:
            final_targets = targets.cpu().numpy()
        else:
            final_targets = np.hstack([final_targets, targets.cpu().numpy()])
        
        #breakpoint()

        # with open('./output/result_adv_features.txt', 'a') as f:
        #     f.write('Epoch: %d, Batch: %d '%(epoch, batch_idx))
        #     f.write('Adv_Loss: %.3f| Adv_Acc: %.3f %% (%d)\n'% (test_adv_loss_1/(batch_idx+1), 
        #                     100.*test_adv_correct_1/total_num,  
        #                     total_num))
    np.save(path+'_features.npy', final_features)
    np.save(path+'_labels.npy', final_targets)
    return

def test_and_save_normal_features(args, net, testloader, device, path): 
    net.eval()

    # loss_fn = torch.nn.CrossEntropyLoss()
    # AutoAttack = LinfAPGDAttack(net, eps=8/255)
    # AutoAttack = L2PGDAttack(net, eps=128/255)

    final_features = None
    final_targets = None

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            final_feature = net.get_final_feature(inputs).cpu().numpy()
        
        if final_features is None:
            final_features = final_feature
        else:
            final_features = np.vstack([final_features, final_feature]) 
        
        if final_targets is None:
            final_targets = targets.cpu().numpy()
        else:
            final_targets = np.hstack([final_targets, targets.cpu().numpy()])
        
        #breakpoint()

        # with open('./output/result_adv_features.txt', 'a') as f:
        #     f.write('Epoch: %d, Batch: %d '%(epoch, batch_idx))
        #     f.write('Adv_Loss: %.3f| Adv_Acc: %.3f %% (%d)\n'% (test_adv_loss_1/(batch_idx+1), 
        #                     100.*test_adv_correct_1/total_num,  
        #                     total_num))
    np.save(path+'_features.npy', final_features)
    np.save(path+'_labels.npy', final_targets)
    return