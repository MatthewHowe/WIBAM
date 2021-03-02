import 

def dets_3D_wcf_to_dets_2D(detections, calib):
  r"""
  Reproject 3D detections in world coordinate frame to 2D bounding boxes
  on all other cameras, trimmed to fit onto frame.
  W.C.F: World coordinate frame
  C.C.F: Camera coordinate frame
  Arguments:
      dets_3D_wcf (np.array, float): list of 3D detections in world.c.f. [[loc,size,rot],...]
      calib (dict): dictionary of calibration information for all cameras
          {P, dist_coefs, rvec, tvec, theta_X_d}
  Returns:
      dets_2D (np.array, float): Detections in 3D on the given camera calibration
          format: 
  """
  BN, objs, dims = detections['location_wcf'].shape
  num_cams = calib['P'].shape[1]
  detections_2D = torch.zeros((BN, num_cams, objs, 4))
  detections_projected3Dbb = torch.zeros((BN, num_cams, objs, 8, 2))
  
  # Convert 3D detections to 3D bounding boxes
  det_3D_to_BBox_3D(detections, calib)

  for batch in range(BN):
    for cam in range(num_cams):
      P = calib['P'][batch][cam]
      rvec = calib['rvec'][batch][cam]
      R_wc = cv2.Rodrigues(rvec.cpu().numpy())[0]
      R_wc = torch.Tensor(R_wc).to(device="cuda")
      tvec = calib['tvec'][batch][cam]

      bounding_box_wcf = detections['3D_bounding_boxes'][batch]
      bounding_box_wcf = bounding_box_wcf.transpose(-2,-1)
      bounding_box_ccf = torch.matmul(R_wc, bounding_box_wcf)
      bounding_box_ccf = torch.add(bounding_box_ccf, tvec)

      bounding_box_cam = torch.matmul(P, bounding_box_ccf)

      bounding_box_cam = torch.div(bounding_box_cam[:],
                                   bounding_box_cam[:,2][:,None,:])

      bounding_box_cam = bounding_box_cam[:,:2].transpose(-2,-1)

      detections_projected3Dbb[batch][cam] = bounding_box_cam

      # Find the minimum rectangle fit around the 3D bounding box
      min_bb = torch.min(bounding_box_cam, axis=1)[0]
      max_bb = torch.max(bounding_box_cam, axis=1)[0]

      # Append result to list
      # TODO: Problem with autograd at this point creating new tensor
      detections_2D[batch,cam,:,0] = min_bb[:,0]
      detections_2D[batch,cam,:,1] = min_bb[:,1]
      detections_2D[batch,cam,:,2] = max_bb[:,0]-min_bb[:,0]
      detections_2D[batch,cam,:,3] = max_bb[:,1]-min_bb[:,1]

  detections['proj_3D_boxes'] = detections_projected3Dbb
  detections['2D_bounding_boxes'] = detections_2D.to(device='cuda')

  return detections