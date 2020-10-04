def Lrec(image_rec, image_orig):

    image_rec_r = image_rec.view(-1,64*64*3)
    image_orig_r = image_orig.view(-1,64*64*3)
    #mse
    rec_loss = torch.norm(image_rec_r - image_orig_r, p=2, dim=1)
    rec_loss = torch.mean(rec_loss)


    return rec_loss


def Lsem(encoder,encoder_rec):

    encoder = encoder.view(-1,1024)
    encoder_rec = encoder_rec.view(-1,1024)

    sem_loss = torch.norm(encoder - encoder_rec, p=1, dim=1)

    sem_loss = torch.mean(sem_loss)

    return sem_loss
