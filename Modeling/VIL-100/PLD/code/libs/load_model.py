import torch
from models.model_s import Model as Model_S
from models.model_c import Model as Model_C
from models.model import Model
from models.loss import *

def load_model_for_test(cfg, dict_DB):
    dict_DB['model_s'] = load_pretrained_model_s(cfg)
    dict_DB['model_c'] = load_pretrained_model_c(cfg)
    if cfg.run_mode == 'test_paper':
        checkpoint = torch.load(f'{cfg.dir["weight_paper"]}/checkpoint_max_F1_vil100_PLD')
    else:
        if cfg.param_name == 'trained_last':
            checkpoint = torch.load(f'{cfg.dir["weight"]}/checkpoint_final')
        elif cfg.param_name == 'max':
            checkpoint = torch.load(f'{cfg.dir["weight"]}/checkpoint_max_F1_{cfg.dataset_name}')
    model = Model(cfg=cfg)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    dict_DB['model'] = model
    return dict_DB

def load_model_for_train(cfg, dict_DB):
    model = Model(cfg=cfg)
    model.cuda()

    dict_DB['model_s'] = load_pretrained_model_s(cfg)
    dict_DB['model_c'] = load_pretrained_model_c(cfg)

    model = Model(cfg=cfg)
    model = load_for_finetuning_pretrained_model(cfg, model)
    model.cuda()
    
    loss_fn = Loss_Function(cfg)
    
    if cfg.optim['mode'] == 'adam_w':
        optimizer = torch.optim.AdamW(params=model.parameters(),
                                      lr=cfg.optim['lr'],
                                      weight_decay=cfg.optim['weight_decay'],
                                    betas=cfg.optim['betas'], eps=cfg.optim['eps'])
        # optimizer = torch.optim.AdamW(
        #                                 [
        #                                     # 主模型参数组
        #                                     {
        #                                         'params': model.parameters(),
        #                                         'lr': cfg.optim['lr'],               # 基础学习率（如 1e-4）
        #                                         'weight_decay': cfg.optim['weight_decay']  # 权重衰减
        #                                     },
                                            
        #                                     # 损失函数权重参数组
        #                                     {
        #                                         'params': loss_fn.coeff_weights,
        #                                         'lr': cfg.optim['lr'] * 0.1,        # 更小的学习率（如 1e-5）
        #                                         'weight_decay': 0.0                 # 可选：禁用权重衰减
        #                                     }
        #                                 ],
        #                                 betas=cfg.optim['betas'],
        #                                 eps=cfg.optim['eps']
        #                             )
    elif cfg.optim['mode'] == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=cfg.optim['lr'],
                                     weight_decay=cfg.optim['weight_decay'])

    cfg.optim['milestones'] = list(np.arange(0, len(dict_DB['trainloader']) * cfg.epochs, len(dict_DB['trainloader']) * 10))[1:]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                     milestones=cfg.optim['milestones'],
                                                     gamma=cfg.optim['gamma'])

    if cfg.resume == True:
        checkpoint = torch.load(f'{cfg.dir["weight"]}/checkpoint_final')
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                         milestones=cfg.optim['milestones'],
                                                         gamma=cfg.optim['gamma'],
                                                         last_epoch=checkpoint['batch_iteration'])
        dict_DB['epoch'] = checkpoint['epoch']
        dict_DB['iteration'] = checkpoint['iteration']
        dict_DB['batch_iteration'] = checkpoint['batch_iteration']
        dict_DB['val_result'] = checkpoint['val_result']

    # loss_fn = Loss_Function(cfg)#####

    dict_DB['model'] = model
    dict_DB['optimizer'] = optimizer
    dict_DB['scheduler'] = scheduler
    dict_DB['loss_fn'] = loss_fn

    return dict_DB

def load_pretrained_model_s(cfg):
    checkpoint = torch.load(f'{cfg.dir["pretrained_weight1"]}/checkpoint_max_seg_fscore_{cfg.dataset_name}')
    # checkpoint = torch.load(f'{cfg.dir["pretrained_weight1"]}/checkpoint_final')
    if cfg.run_mode == 'test_paper':
        checkpoint = torch.load(f'{cfg.dir["weight_paper"]}/checkpoint_max_seg_fscore_vil100_ILD_seg')

    model = Model_S(cfg=cfg)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    return model

def load_pretrained_model_c(cfg):
    checkpoint = torch.load(f'{cfg.dir["pretrained_weight2"]}/checkpoint_max_F1_{cfg.dataset_name}')
    # checkpoint = torch.load(f'{cfg.dir["pretrained_weight2"]}/checkpoint_final')
    if cfg.run_mode == 'test_paper':
        checkpoint = torch.load(f'{cfg.dir["weight_paper"]}/checkpoint_max_F1_vil100_ILD_coeff')
    model = Model_C(cfg=cfg)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    return model

def load_for_finetuning_pretrained_model(cfg, model):
    checkpoint1 = torch.load(f'{cfg.dir["pretrained_weight1"]}/checkpoint_max_seg_fscore_{cfg.dataset_name}')
    checkpoint2 = torch.load(f'{cfg.dir["pretrained_weight2"]}/checkpoint_max_F1_{cfg.dataset_name}')
    # checkpoint1 = torch.load(f'{cfg.dir["pretrained_weight1"]}/checkpoint_final')
    # checkpoint2 = torch.load(f'{cfg.dir["pretrained_weight2"]}/checkpoint_final')
    for param in list(checkpoint1['model']):
        if 'classifier' not in param:
            del checkpoint1['model'][param]

    model.load_state_dict(checkpoint1['model'], strict=False)
    model.load_state_dict(checkpoint2['model'], strict=False)
    model.cuda()
    return model
