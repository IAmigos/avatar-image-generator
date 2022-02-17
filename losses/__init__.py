import torch
from .utils_loss import calc_eval_stats, calc_fid

def L2_norm(image_rec, image_orig):

  assert image_rec.shape == image_orig.shape, "Assertion error: shape of input should be as same as target"
  
  return torch.linalg.norm(image_rec.reshape(image_rec.shape[0], -1) - image_orig.reshape(image_orig.shape[0], -1), ord=2, dim=1).mean()



def L1_norm(encoder,encoder_rec):

  assert encoder.shape == encoder_rec.shape, "Assertion error: shape of input should be as same as target"
  
  return torch.linalg.norm(encoder.reshape(encoder.shape[0], -1) - encoder_rec.reshape(encoder_rec.shape[0], -1), ord=1, dim=1).mean()

def fid(real_img,fake_img,model, device):
  #Code obtained from: https://www.kaggle.com/ibtesama/gan-in-pytorch-with-fid
  mu_1,std_1=calc_eval_stats(real_img,model,device=device)
  mu_2,std_2=calc_eval_stats(fake_img,model,device=device)
    
  """get fretched distance"""
  fid_value = calc_fid(mu_1, std_1, mu_2, std_2)
  return fid_value

def MMD(x, y, kernel, device):
    #Code obtained from: https://www.kaggle.com/onurtunali/maximum-mean-discrepancy
    """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
    return torch.mean(XX + YY - 2. * XY).item()