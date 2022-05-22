import torch
import torch.nn as nn
import torch.optim as optim
from models import *
from model_train import net_sample_output, train_net
from model_evaluate import visualize_output

if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    net = ResNet18(48).to(device)
    net.load_state_dict(torch.load('model_keypoints_24pts_iter_2.pt'))
    net.eval()

    # get a sample of test data
    with torch.no_grad():
        test_images, test_outputs, gt_pts = net_sample_output(net, device)
        visualize_output(test_images, test_outputs,gt_pts)

