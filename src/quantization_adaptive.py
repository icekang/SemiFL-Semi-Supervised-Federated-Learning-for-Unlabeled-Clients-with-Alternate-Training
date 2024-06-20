from config import cfg, process_args
import models
from utils import save, to_device, process_control, process_dataset, resume, collate
from data import fetch_dataset, make_data_loader, make_batchnorm_stats
import os

from metrics import Metric
import torch
from logger import make_logger

data = None # global variable to store data for quantization

def quantize_model(model_fp32):
    model_fp32.eval()
    model_fp32_prepared = torch.ao.quantization.prepare_qat(model_fp32.train())
    # set quantization config for server (x86)
    model_fp32.qconfig = torch.quantization.get_default_qconfig('x86')
    model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)
    input_fp32 = data
    model_fp32_prepared(input_fp32)
    model_fp32_prepared.to('cpu')
    model_fp32_prepared.eval()

    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
    return model_int8

def runExperiment():
    global model, data_loader
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    metric = Metric({'test': ['Loss', 'Accuracy']})
    result = resume(cfg['model_tag'], load_tag='best')
    last_epoch = result['epoch']
    supervised_idx = result['supervised_idx']
    data_split = result['data_split']
    model.load_state_dict(result['server'].model_state_dict)
    data_loader = make_data_loader(dataset, 'server')
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    test_logger.safe(True)
    test_model = make_batchnorm_stats(dataset['train'], model, 'server')
    model_size(test_model.state_dict())
    torch.save(test_model.state_dict(), 'model_{}.pt'.format(cfg['model_tag']))
    test_model.to('cpu')
    cfg['device'] = 'cpu'
    print('Before quantization:')
    test(data_loader['test'], test_model, metric, test_logger, last_epoch, profiler_suffix='before_quantization')
    
    test_model.to('cuda:0')
    cfg['device'] = 'cuda:0'
    test_model = quantize_model(test_model)
    torch.save(test_model.state_dict(), 'quantized_model_{}.pt'.format(cfg['model_tag']))
    model_size(test_model.state_dict())
    test_model.to('cpu')
    cfg['device'] = 'cpu'
    print('After quantization:')
    test(data_loader['test'], test_model, metric, test_logger, last_epoch, profiler_suffix='after_quantization')
    test_logger.safe(False)
    result = resume(cfg['model_tag'], load_tag='checkpoint')
    train_logger = result['logger'] if 'logger' in result else None
    result = {'cfg': cfg, 'epoch': last_epoch, 'supervised_idx': supervised_idx, 'data_split': data_split,
              'logger': {'train': train_logger, 'test': test_logger}}
    save(result, './output/result/{}.pt'.format(cfg['model_tag']))
    return

def test(data_loader, model, metric, logger, epoch, profiler_suffix=''):
    print('Test Epoch: {}'.format(epoch))
    from tqdm import tqdm
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1),
        profile_memory=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/adaptive_{profiler_suffix}'),
        with_stack=True
    ) as profiler:
        with torch.no_grad():
            model.train(False)
            for i, input in tqdm(enumerate(data_loader), total=len(data_loader), desc='Testing'):
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                output = model(input)
                output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                logger.append(evaluation, 'test', input_size)
                profiler.step()
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
            logger.append(info, 'test', mean=False)
            print(logger.write('test', metric.metric_name['test']), flush=True)
    return

def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return

def model_size(model_state_dict):
    size_model = 0
    for param in model_state_dict.values():
        if param is None:
            continue

        if isinstance(param, torch.dtype):
            # qint8 quantized model
            size_model += torch.iinfo(param).bits

        if isinstance(param, torch.Tensor):
            if torch.is_floating_point(param):
                size_model += param.numel() * torch.finfo(param.dtype).bits
            else:
                size_model += param.numel() * torch.iinfo(param.dtype).bits

    print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")


if __name__ == '__main__':
    args = {
        'data_name': 'CIFAR10',
        'model_name': 'wresnet28x2',
        'control_name': '250_fix@0.95_5_1_non-iid-d-0.3_5-5_0.5_0_1',
        'config_file': 'config_adaptive.yml',
        'output_dir': 'output_adaptive/'
    }

    for k in cfg:
        args[k] = cfg[k]
    process_args(args)
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        cfg['seed'] = int(cfg['model_tag'].split('_')[0])
        print('Experiment: {}'.format(cfg['model_tag']))

    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)

    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    result = resume(cfg['model_tag'], load_tag='best')
    model.load_state_dict(result['server'].model_state_dict)

    data_loader = make_data_loader(dataset, 'server')

    data = next(iter(data_loader['test']))
    data = collate(data)
    data = to_device(data, cfg['device'])

    main()