from models import *
import torch


def test(weights_path, img_path, use_gpu):
    device = torch.device(
        "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    e1 = Encoder()
    e_shared = Eshared(0.5)
    d_shared = Dshared()
    d2 = Decoder()

    e1.load_state_dict(torch.load(weights_path + 'e1.pth'))
    e_shared.load_state_dict(torch.load(
        weights_path + 'e_shared.pth'))
    d_shared.load_state_dict(torch.load(
        weights_path + 'd_shared.pth'))
    d2.load_state_dict(torch.load(weights_path + 'd2.pth'))

    e1.to(device)
    e_shared.to(device)
    d_shared.to(device)
    d2.to(device)

    e1.eval()
    e_shared.eval()
    d_shared.eval()
    d2.eval()
