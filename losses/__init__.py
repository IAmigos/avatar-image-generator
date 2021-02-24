import torch

def L2_norm(image_rec, image_orig):

  assert image_rec.shape == image_orig.shape, "Assertion error: shape of input should be as same as target"
  
  return torch.linalg.norm(image_rec.reshape(image_rec.shape[0], -1) - image_orig.reshape(image_orig.shape[0], -1), ord=2, dim=1).mean()

#  image_rec_r = image_rec.view(-1,64*64*3)
#  image_orig_r = image_orig.view(-1,64*64*3)
  
#  rec_loss = torch.norm(image_rec_r - image_orig_r, p=2, dim=1)
#  rec_loss = torch.mean(rec_loss)


#  return rec_loss


def L1_norm(encoder,encoder_rec):

  assert encoder.shape == encoder_rec.shape, "Assertion error: shape of input should be as same as target"
  
  return torch.linalg.norm(encoder.reshape(encoder.shape[0], -1) - encoder_rec.reshape(encoder_rec.shape[0], -1), ord=1, dim=1).mean()



#  encoder = encoder.view(-1,1024)
#  encoder_rec = encoder_rec.view(-1,1024)

#  sem_loss = torch.norm(encoder - encoder_rec, p=1, dim=1)
  
#  sem_loss = torch.mean(sem_loss)

#  return sem_loss 
