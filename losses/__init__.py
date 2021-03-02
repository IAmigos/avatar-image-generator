import torch

def L2_norm(image_rec, image_orig):

  assert image_rec.shape == image_orig.shape, "Assertion error: shape of input should be as same as target"
  
  return torch.linalg.norm(image_rec.reshape(image_rec.shape[0], -1) - image_orig.reshape(image_orig.shape[0], -1), ord=2, dim=1).mean()



def L1_norm(encoder,encoder_rec):

  assert encoder.shape == encoder_rec.shape, "Assertion error: shape of input should be as same as target"
  
  return torch.linalg.norm(encoder.reshape(encoder.shape[0], -1) - encoder_rec.reshape(encoder_rec.shape[0], -1), ord=1, dim=1).mean()

