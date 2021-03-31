from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import torch
import numpy as np
from progress.bar import Bar

from model.data_parallel import DataParallel
from utils.utils import AverageMeter

from model.losses import FastFocalLoss, RegWeightedL1Loss
from model.losses import BinRotLoss, WeightedBCELoss
from model.mv_losses import ReprojectionLoss
from model.decode import generic_decode
from model.utils import _sigmoid, flip_tensor, flip_lr_off, flip_lr
from utils.debugger import Debugger
from utils.post_process import generic_post_process
from utils.utils import Profiler

class GenericLoss(torch.nn.Module):
  def __init__(self, opt):
    super(GenericLoss, self).__init__()
    self.crit = FastFocalLoss(opt=opt)
    self.crit_reg = RegWeightedL1Loss()
    if 'rot' in opt.heads:
      self.crit_rot = BinRotLoss()
    if 'nuscenes_att' in opt.heads:
      self.crit_nuscenes_att = WeightedBCELoss()
    self.opt = opt

  def _sigmoid_output(self, output):
    if 'hm' in output:
      output['hm'] = _sigmoid(output['hm'])
    if 'hm_hp' in output:
      output['hm_hp'] = _sigmoid(output['hm_hp'])
    if 'dep' in output:
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
    return output

  def forward(self, outputs, batch):
    opt = self.opt
    # reset all losses to zero
    losses = {head: 0 for head in opt.heads}


    output = outputs
    output = self._sigmoid_output(output)

    # Heatmap loss
    if 'hm' in output:
      losses['hm'] += self.crit(
        output['hm'], batch['hm'], batch['ind'],
        batch['mask'], batch['cat']) / opt.num_stacks

    regression_heads = [
      'reg', 'wh', 'tracking', 'ltrb', 'ltrb_amodal', 'hps',
      'dep', 'dim', 'amodel_offset', 'velocity']

    for head in regression_heads:
      if head in output:
        losses[head] += self.crit_reg(
          output[head], batch[head + '_mask'],
          batch['ind'], batch[head]) / opt.num_stacks

    if 'hm_hp' in output:
      losses['hm_hp'] += self.crit(
        output['hm_hp'], batch['hm_hp'], batch['hp_ind'],
        batch['hm_hp_mask'], batch['joint']) / opt.num_stacks
      if 'hp_offset' in output:
        losses['hp_offset'] += self.crit_reg(
          output['hp_offset'], batch['hp_offset_mask'],
          batch['hp_ind'], batch['hp_offset']) / opt.num_stacks

    if 'rot' in output:
      losses['rot'] += self.crit_rot(
        output['rot'], batch['rot_mask'], batch['ind'], batch['rotbin'],
        batch['rotres']) / opt.num_stacks

    if 'nuscenes_att' in output:
      losses['nuscenes_att'] += self.crit_nuscenes_att(
        output['nuscenes_att'], batch['nuscenes_att_mask'],
        batch['ind'], batch['nuscenes_att']) / opt.num_stacks

    losses['tot'] = 0
    for head in opt.heads:
      losses['tot'] += opt.weights[head] * losses[head]
    
    
    
    return losses['tot'], losses

class MultiviewLoss(torch.nn.Module):
  r"""
  Class for calculating the multi-view losses for the WIBAM dataset
  Arguments:
      opt (dict): Configuration options
  """
  def __init__(self, opt=None):
    super(MultiviewLoss, self).__init__()
    self.FastFocalLoss = FastFocalLoss(opt=opt)
    self.RegWeightedL1Loss = RegWeightedL1Loss()
    self.ReprojectionLoss = ReprojectionLoss(opt)
    self.opt = opt
    self.loss_stats = {}

  def _sigmoid_output(self, output):
    if 'hm' in output:
      output['hm'] = _sigmoid(output['hm'])
    if 'hm_hp' in output:
      output['hm_hp'] = _sigmoid(output['hm_hp'])
    if 'dep' in output:
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
      output['dep'] *= self.opt.depth_scale
    return output

  def forward(self, outputs, batch):
    r"""
    Class for calculating the multi-view losses for the WIBAM dataset
    Arguments:
      outputs (dict): Output from the detection model on given batch
      batch (dict): The input information for this batch
    Returns:
      losses (float): total loss
      losses (list): list of individual losses
    """
    opt = self.opt
    # reset all losses to zero
    if self.opt.mv_only:
      losses = {'mv':0, 'tot':0}
    else:
      losses = {'hm':0, 'reg':0, 'wh':0, 'mv':0, 'tot':0}
    
    if isinstance(outputs, list):
      output = outputs[0]
    else:
      output = outputs
    output = self._sigmoid_output(output)

    if not self.opt.mv_only:
      regression_heads = ['reg', 'wh']

      for head in regression_heads:
        if head in output:
          losses[head] += self.RegWeightedL1Loss(
            output[head], batch[head + '_mask'],
            batch['ind'], batch[head]) / opt.num_stacks

      # Heatmap loss
      if 'hm' in output:
        losses['hm'] += self.FastFocalLoss(
          output['hm'], batch['hm'], batch['ind'],
          batch['mask_det'], batch['cat_det']) / opt.num_stacks

      # Reprojection loss
      mv_loss = self.ReprojectionLoss(output,batch)
      if 'tot' not in mv_loss:
        losses['mv'] += mv_loss['tot']

      for key, val in losses.items():
        if key != 'tot':
          losses['tot'] += val * self.opt.weights[key]

      for key, val in mv_loss.items():
        if key != 'tot':
          losses["mv_{}".format(key)] = val

      return losses['tot'], losses

class LossStats():
  def __init__(self):
    self.loss_stats = {}

class ModleWithLoss(torch.nn.Module):
  def __init__(self, model, loss, LStats):
    super(ModleWithLoss, self).__init__()
    self.model = model
    self.loss = loss
    self.LStats = LStats

  def forward(self, batch, profiler=False):
    pre_img = batch['pre_img'] if 'pre_img' in batch else None
    pre_hm = batch['pre_hm'] if 'pre_hm' in batch else None
    
    outputs = self.model(batch['image'], pre_img, pre_hm)
    if profiler:
      profiler.interval_trigger("Run model")
    loss, loss_stats = self.loss(outputs, batch)
    if profiler:
      profiler.interval_trigger("Calculate loss")
    for key, val in loss_stats.items():
      if val == 0:
        continue
      loss_stats[key] = val.detach()
    self.LStats.loss_stats = loss_stats
    return outputs[-1], loss

class Trainer(object):
  def __init__(
    self, opt, model, writer, total_writer, optimizer=None, dataset=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt, dataset)
    if dataset is not None:
      self.dataset = dataset
    else:
      self.dataset = opt.dataset
    self.LStats = LossStats()
    self.model_with_loss = ModleWithLoss(model, self.loss, self.LStats)
    self.writer = writer
    self.total_writer = total_writer
    self.total_steps_train = 0
    self.total_steps_val = 0

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      self.model_with_loss = DataParallel(
        self.model_with_loss, device_ids=gpus,
        chunk_sizes=chunk_sizes).to(device)
    else:
      self.model_with_loss = self.model_with_loss.to(device)

    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader):
    # torch.autograd.set_detect_anomaly(True)
    model_with_loss = self.model_with_loss
    if phase == 'train':

      model_with_loss.train()
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = None
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    bar.width = 5
    end = time.time()
    self.profiler = Profiler()
    self.profiler.start()

    # Training loop
    for iter_id, batch in enumerate(data_loader):
      self.profiler.interval_trigger("Load data")
      #  Break if epoch complete
      if iter_id >= num_iters:
        break

      data_time.update(time.time() - end)
      end = time.time()

      # Put batches to GPU
      for k in batch:
        if k != 'meta' and k != 'calib' and k != 'drawing_images'\
            and k != 'image_id':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)

      self.profiler.interval_trigger("Data to GPU")

      # Run outputs for batch from model with losses
      # Loss is the total loss for the batch
      output, loss = model_with_loss(batch, self.profiler)
      loss_stats = self.LStats.loss_stats

      # self.profiler.interval_trigger("Run model")

      # If training phase, back propogate the loss
      if phase == 'train':
        loss = loss.mean()
        self.optimizer.zero_grad()
        
        loss.backward()
        self.profiler.interval_trigger("Back prop")
        self.optimizer.step()
        self.profiler.interval_trigger("Optimiser step")
      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)

      if avg_loss_stats is None:
        avg_loss_stats = {l: AverageMeter() for l in loss_stats}

      # Logging step
      for l in loss_stats:
        try:
          loss_val = torch.mean(loss_stats[l]).item()
        except:
          continue
        if l not in avg_loss_stats:
          avg_loss_stats[l] = AverageMeter()
        if loss_val != 0:
          avg_loss_stats[l].update(
            loss_val, batch['image'].size(0)
          )

          # Tensorboard log
          if l == "amodel_offset":
            continue
          else:
            if phase == "train":
              self.writer.add_scalar("{}: {}_{}".format(self.dataset,l,phase), avg_loss_stats[l].val, self.total_steps_train)
            elif phase == "val":
              self.writer.add_scalar("{}: {}_{}".format(self.dataset,l,phase), avg_loss_stats[l].val, self.total_steps_val)

      for l, val in avg_loss_stats.items():
        Bar.suffix = Bar.suffix + '|{} {:.2f} '.format(l, avg_loss_stats[l].avg)
      Bar.suffix = Bar.suffix + 'Data {dt.avg:.3f}s |Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0: # If not using progress bar
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
          print('\n')
      else:
        bar.next()
        print(' ')

      #
      if opt.debug > 0:
        self.debug(batch, output, iter_id, dataset=data_loader.dataset)

      # Iterate step
      if phase == "train":
        self.total_steps_train += 1
      elif phase == "val":
        self.total_steps_val += 1

      self.profiler.interval_trigger("Finish")

      del output, loss, loss_stats
      # self.profiler.print_interval_times()
      self.profiler.start()
      torch.cuda.empty_cache()

    #
    bar.finish()
    for loss, value in avg_loss_stats.items():
      self.total_writer.add_scalar("EPOCH AV {}: {}_{}".format(self.dataset,phase, loss), value.avg, epoch)
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results
  
  def _get_losses(self, opt, dataset=None):
    if dataset is None:
      dataset=opt.dataset
    if dataset == "wibam":
      loss_order = ['hm', 'wh', 'reg']
      loss_states = ['tot','mv'] + [i for i in loss_order if i in opt.heads]
      loss = MultiviewLoss(opt)
    else:
      loss_order = ['hm', 'wh', 'reg', 'ltrb', 'hps', 'hm_hp', \
        'hp_offset', 'dep', 'dim', 'rot', 'amodel_offset', \
        'ltrb_amodal', 'tracking', 'nuscenes_att', 'velocity']
      loss_states = ['tot'] + [k for k in loss_order if k in opt.heads]
      loss = GenericLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id, dataset):
    opt = self.opt
    if 'pre_hm' in batch:
      output.update({'pre_hm': batch['pre_hm']})
    dets = generic_decode(output, K=opt.K, opt=opt)
    for k in dets:
      dets[k] = dets[k].detach().cpu().numpy()
    dets_gt = batch['meta']['gt_det']
    for i in range(1):
      debugger = Debugger(opt=opt, dataset=dataset)
      img = batch['image'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * dataset.std + dataset.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')

      if 'pre_img' in batch:
        pre_img = batch['pre_img'][i].detach().cpu().numpy().transpose(1, 2, 0)
        pre_img = np.clip(((
          pre_img * dataset.std + dataset.mean) * 255), 0, 255).astype(np.uint8)
        debugger.add_img(pre_img, 'pre_img_pred')
        debugger.add_img(pre_img, 'pre_img_gt')
        if 'pre_hm' in batch:
          pre_hm = debugger.gen_colormap(
            batch['pre_hm'][i].detach().cpu().numpy())
          debugger.add_blend_img(pre_img, pre_hm, 'pre_hm')

      debugger.add_img(img, img_id='out_pred')
      if 'ltrb_amodal' in opt.heads:
        debugger.add_img(img, img_id='out_pred_amodal')
        debugger.add_img(img, img_id='out_gt_amodal')

      # Predictions
      for k in range(len(dets['scores'][i])):
        if dets['scores'][i, k] > opt.vis_thresh:
          debugger.add_coco_bbox(
            dets['bboxes'][i, k] * opt.down_ratio, dets['clses'][i, k],
            dets['scores'][i, k], img_id='out_pred')

          if 'ltrb_amodal' in opt.heads:
            debugger.add_coco_bbox(
              dets['bboxes_amodal'][i, k] * opt.down_ratio, dets['clses'][i, k],
              dets['scores'][i, k], img_id='out_pred_amodal')

          if 'hps' in opt.heads and int(dets['clses'][i, k]) == 0:
            debugger.add_coco_hp(
              dets['hps'][i, k] * opt.down_ratio, img_id='out_pred')

          if 'tracking' in opt.heads:
            debugger.add_arrow(
              dets['cts'][i][k] * opt.down_ratio, 
              dets['tracking'][i][k] * opt.down_ratio, img_id='out_pred')
            debugger.add_arrow(
              dets['cts'][i][k] * opt.down_ratio, 
              dets['tracking'][i][k] * opt.down_ratio, img_id='pre_img_pred')

      # Ground truth
      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt['scores'][i])):
        if dets_gt['scores'][i][k] > opt.vis_thresh:
          debugger.add_coco_bbox(
            dets_gt['bboxes'][i][k] * opt.down_ratio, dets_gt['clses'][i][k],
            dets_gt['scores'][i][k], img_id='out_gt')

          if 'ltrb_amodal' in opt.heads:
            debugger.add_coco_bbox(
              dets_gt['bboxes_amodal'][i, k] * opt.down_ratio, 
              dets_gt['clses'][i, k],
              dets_gt['scores'][i, k], img_id='out_gt_amodal')

          if 'hps' in opt.heads and \
            (int(dets['clses'][i, k]) == 0):
            debugger.add_coco_hp(
              dets_gt['hps'][i][k] * opt.down_ratio, img_id='out_gt')

          if 'tracking' in opt.heads:
            debugger.add_arrow(
              dets_gt['cts'][i][k] * opt.down_ratio, 
              dets_gt['tracking'][i][k] * opt.down_ratio, img_id='out_gt')
            debugger.add_arrow(
              dets_gt['cts'][i][k] * opt.down_ratio, 
              dets_gt['tracking'][i][k] * opt.down_ratio, img_id='pre_img_gt')

      if 'hm_hp' in opt.heads:
        pred = debugger.gen_colormap_hp(
          output['hm_hp'][i].detach().cpu().numpy())
        gt = debugger.gen_colormap_hp(batch['hm_hp'][i].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hmhp')
        debugger.add_blend_img(img, gt, 'gt_hmhp')


      if 'rot' in opt.heads and 'dim' in opt.heads and 'dep' in opt.heads:
        dets_gt = {k: dets_gt[k].cpu().numpy() for k in dets_gt}
        calib = batch['meta']['calib'].detach().numpy() \
                if 'calib' in batch['meta'] else None
        det_pred = generic_post_process(opt, dets, 
          batch['meta']['c'].cpu().numpy(), batch['meta']['s'].cpu().numpy(),
          output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
          calib)
        det_gt = generic_post_process(opt, dets_gt, 
          batch['meta']['c'].cpu().numpy(), batch['meta']['s'].cpu().numpy(),
          output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
          calib)

        debugger.add_3d_detection(
          batch['meta']['img_path'][i], batch['meta']['flipped'][i],
          det_pred[i], calib[i],
          vis_thresh=opt.vis_thresh, img_id='add_pred')
        debugger.add_3d_detection(
          batch['meta']['img_path'][i], batch['meta']['flipped'][i], 
          det_gt[i], calib[i],
          vis_thresh=opt.vis_thresh, img_id='add_gt')
        debugger.add_bird_views(det_pred[i], det_gt[i], 
          vis_thresh=opt.vis_thresh, img_id='bird_pred_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)
