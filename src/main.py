from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os

from pathlib import Path
import torch
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from opts import opts
from model.model import create_model, load_model, save_model
from model.data_parallel import DataParallel
from logger import Logger
from utils.collate import default_collate, instance_batching_collate
from dataset.dataset_factory import get_dataset
from trainer import Trainer

from utils.net import *

def get_optimizer(opt, model):
  if opt.optim == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  elif opt.optim == 'sgd':
    print('Using SGD')
    optimizer = torch.optim.SGD(
      model.parameters(), opt.lr, momentum=0.9, weight_decay=0.0001)
  else:
    assert 0, opt.optim
  return optimizer

def main(opt):
  time_str = time.strftime('%Y-%m-%d-%H-%M')
  tensorboard_dir = opt.save_dir + '/tensorboard_{}'.format(time_str)
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

  # Get dataset name
  Dataset = get_dataset(opt.dataset)

  # Initialise training task
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  if not opt.not_set_cuda_env:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

  # Initialise loggers
  logger = Logger(opt)
  writer = SummaryWriter(tensorboard_dir)
  total_writer = SummaryWriter(tensorboard_dir)

  # Create model
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)

  # Initialise optimiser
  optimizer = get_optimizer(opt, model)
  start_epoch = 0

  # Load model from options
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, opt, optimizer)

  
  # Initialise trainer class
  trainer = Trainer(opt, model, writer, total_writer, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  # If validation during traininer or running test initialise
  if opt.val_intervals < opt.num_epochs or opt.test:
    print('Setting up validation data...')
    val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), batch_size=1, shuffle=True, num_workers=1,
      pin_memory=True)

    if opt.test:
      _, preds = trainer.val(0, val_loader)
      val_loader.dataset.run_eval(preds, opt.save_dir)
      return

  # Initialising trianing dataset
  print('Setting up train data...')

  if opt.instance_batching:
    if opt.batch_size % 4 == 0:
      batch_size = int(opt.batch_size/4)
    else:   
      print("[ERROR] Batch size must be divisible by num cams")
      end()
    collate_fn=instance_batching_collate
  else:
    batch_size = opt.batch_size
    collate_fn=default_collate

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), batch_size=batch_size, shuffle=True,
      num_workers=opt.num_workers, 
      collate_fn=collate_fn, 
      pin_memory=True, drop_last=True
  )

  print('Starting training...')
  for epoch in range(start_epoch , opt.num_epochs + 1):
    
    mark = epoch if opt.save_all else 'last'
    

    
    # Complete training step
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))

    # Write log
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    
    # Validation
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)     
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
        if opt.eval_val:
          val_loader.dataset.run_eval(preds, opt.save_dir)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
    # Save the model
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)

    logger.write('\n')
    if epoch in opt.save_point:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      
    if epoch in opt.lr_step:
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
    if opt.gs_sync:
      gsutil_sync(True, "aiml-reid-casr-data", Path(opt.save_dir),
                  "", bucket_prefix_folder="wibam_experiments")

  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
