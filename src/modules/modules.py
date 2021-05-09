import copy
import datetime
import numpy as np
import sys
import time
import torch
import torch.nn.functional as F
import models
from itertools import compress
from config import cfg
from data import make_data_loader, make_batchnorm_stats, MixDataset
from utils import to_device, make_optimizer, collate
from metrics import Accuracy


class Server:
    def __init__(self, model):
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        optimizer = make_optimizer(model, 'local')
        self.optimizer_state_dict = optimizer.state_dict()
        global_optimizer = make_optimizer(model, 'global')
        self.global_optimizer_state_dict = global_optimizer.state_dict()

    def distribute(self, dataset, client):
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(self.model_state_dict)
        model = make_batchnorm_stats(dataset, model, 'global')
        model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        for m in range(len(client)):
            if client[m].active:
                client[m].model_state_dict = copy.deepcopy(model_state_dict)
        return

    def update(self, client):
        with torch.no_grad():
            valid_client = [client[i] for i in range(len(client)) if client[i].active]
            if len(valid_client) > 0:
                model = eval('models.{}()'.format(cfg['model_name']))
                model.load_state_dict(self.model_state_dict)
                global_optimizer = make_optimizer(model, 'global')
                global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                global_optimizer.zero_grad()
                weight = torch.ones(len(valid_client))
                weight = weight / weight.sum()
                for k, v in model.named_parameters():
                    parameter_type = k.split('.')[-1]
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        tmp_v = v.data.new_zeros(v.size())
                        for m in range(len(valid_client)):
                            tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                        v.grad = (v.data - tmp_v).detach()
                global_optimizer.step()
                self.global_optimizer_state_dict = global_optimizer.state_dict()
                self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            for i in range(len(client)):
                client[i].active = False
        return

    def train(self, dataset, lr, metric, logger):
        data_loader = make_data_loader({'train': dataset}, 'server')['train']
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(self.model_state_dict)
        self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer = make_optimizer(model, 'local')
        optimizer.load_state_dict(self.optimizer_state_dict)
        model.train(True)
        for epoch in range(1, cfg['local']['num_epochs'] + 1):
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                logger.append(evaluation, 'train', n=input_size)
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        return


class Client:
    def __init__(self, client_id, model, data_split):
        self.client_id = client_id
        self.data_split = data_split
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        optimizer = make_optimizer(model, 'local')
        self.optimizer_state_dict = optimizer.state_dict()
        self.active = False
        self.buffer = None
        self.beta = torch.distributions.beta.Beta(torch.tensor([cfg['alpha']]), torch.tensor([cfg['alpha']]))
        self.verbose = False

    def make_hard_pseudo_label(self, soft_pseudo_label):
        max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_p.ge(cfg['threshold'])
        return hard_pseudo_label, mask

    def make_weight(self, target):
        cls_indx, cls_counts = torch.unique(target, return_counts=True)
        num_samples_per_cls = torch.zeros(cfg['target_size'], dtype=torch.float32)
        num_samples_per_cls[cls_indx] = cls_counts.float()
        beta = torch.tensor(0.999, dtype=torch.float32)
        effective_num = 1.0 - beta.pow(num_samples_per_cls)
        weight = (1.0 - beta) / effective_num
        weight[torch.isinf(weight)] = 0
        weight = weight / torch.sum(weight) * (weight > 0).float().sum()
        return weight

    def make_dataset(self, dataset):
        if 'sup' in cfg['loss_mode']:
            self.weight = self.make_weight(torch.tensor(dataset.target)) if cfg['weight'] else None
            return dataset
        elif 'fix' in cfg['loss_mode']:
            with torch.no_grad():
                data_loader = make_data_loader({'train': dataset}, 'global', shuffle={'train': False})['train']
                model = eval('models.{}(track=True).to(cfg["device"])'.format(cfg['model_name']))
                model.load_state_dict(self.model_state_dict)
                model.train(False)
                output = []
                target = []
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input = to_device(input, cfg['device'])
                    _output = model(input)
                    output_i = _output['target']
                    target_i = input['target']
                    output.append(output_i.cpu())
                    target.append(target_i.cpu())
                output = torch.cat(output, dim=0)
                target = torch.cat(target, dim=0)
                self.buffer = F.softmax(output, dim=-1)
                acc = Accuracy(self.buffer, target)
                new_target, mask = self.make_hard_pseudo_label(self.buffer)
                if torch.all(~mask):
                    if self.verbose:
                        print('Model: {} Accuracy: {:.3f}, Number of Labeled: 0({})'.format(cfg['model_tag'], acc,
                                                                                            len(output)))
                    return None
                else:
                    self.weight = self.make_weight(new_target) if cfg['weight'] else None
                    new_acc = Accuracy(self.buffer[mask], target[mask])
                    num_labeled = int(mask.float().sum())
                    if self.verbose:
                        print('Model: {} Accuracy: {:.3f} ({:.3f}), Number of Labeled: {}({})'.format(cfg['model_tag'],
                                                                                                      acc, new_acc,
                                                                                                      num_labeled,
                                                                                                      len(output)))
                    fix_dataset = copy.deepcopy(dataset)
                    fix_dataset.target = new_target.tolist()
                    mask = mask.tolist()
                    fix_dataset.data = list(compress(fix_dataset.data, mask))
                    fix_dataset.target = list(compress(fix_dataset.target, mask))
                    fix_dataset.other = {'id': list(range(len(fix_dataset.data)))}
                    if 'mix' in cfg['loss_mode']:
                        mix_dataset = copy.deepcopy(dataset)
                        mix_dataset.target = new_target.tolist()
                        mix_dataset = MixDataset(len(fix_dataset), mix_dataset)
                    else:
                        mix_dataset = None
                    return fix_dataset, mix_dataset
        else:
            raise ValueError('Not valid client loss mode')
        return

    def train(self, dataset, lr, metric, logger):
        if cfg['loss_mode'] == 'sup':
            data_loader = make_data_loader({'train': dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model, 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            for epoch in range(1, cfg['local']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['weight'] = self.weight
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
        elif cfg['loss_mode'] == 'fix':
            fix_dataset, _ = dataset
            data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model, 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            for epoch in range(1, cfg['local']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['weight'] = self.weight
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
        elif cfg['loss_mode'] == 'fix-mix':
            fix_dataset, mix_dataset = dataset
            fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
            mix_data_loader = make_data_loader({'train': mix_dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model, 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            for epoch in range(1, cfg['local']['num_epochs'] + 1):
                for i, (fix_input, mix_input) in enumerate(zip(fix_data_loader, mix_data_loader)):
                    input = {'data': fix_input['data'], 'target': fix_input['target'], 'aug': fix_input['aug'],
                             'mix_data': mix_input['data'], 'mix_target': mix_input['target']}
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['lam'] = self.beta.sample()[0]
                    input['mix_data'] = (input['lam'] * input['data'] + (1 - input['lam']) * input['mix_data']).detach()
                    input['mix_target'] = torch.stack([input['target'], input['mix_target']], dim=-1)
                    input['weight'] = self.weight
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
        else:
            raise ValueError('Not valid client loss mode')
        self.optimizer_state_dict = optimizer.state_dict()
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        return
