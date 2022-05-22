import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_load import FacialKeypointsDataset,FaceLandmarksDataset
from data_load import Rescale, RandomCrop, Normalize, ToTensor
from models import *
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from datetime import datetime


use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 4} if torch.cuda.is_available() else {}
device = torch.device("cuda" if use_cuda else "cpu")

net = ResNet18(48).to(device)
net.load_state_dict(torch.load('model_keypoints_24pts_iter_2.pt'))
net.eval()

if __name__ =='__main__' :

    print('Evaluation(MSE)')
    s = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(s)

    data_transform = transforms.Compose([Rescale(250),
                                         RandomCrop(224),
                                         Normalize(),
                                         ToTensor()])

    # load test data in batches
    batch_size = 1

    # create the test dataset
    test_dataset = FacialKeypointsDataset(csv_file='test.csv', 
                                          root_dir='./data/test/',
                                          transform=data_transform)

    # load test data in batches
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=True,
                             **kwargs)

    a = 0
    RMSE_avg = 0
    RMSE_total = 0
    total_num = 0

    for data in test_loader:
        images = data['image']
        images = Variable(images)
        name = data['image_name']
        gt_pts = data['keypoints']

         # convert images to FloatTensors
        if (torch.cuda.is_available()):
            images = images.type(torch.cuda.FloatTensor)
            images.to(device)
        else:
            images = images.type(torch.FloatTensor)

        # forward pass to get net output
        output_pts = net(images)

        a += 1
        print(a)
        s = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(s)

        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], -1, 2)

        for i in range(batch_size):
            #print(i)
            # un-transform the predicted key_pts data
            predicted_key_pts = output_pts[i].data
            if (torch.cuda.is_available()):
                predicted_key_pts = predicted_key_pts.cpu()

            predicted_key_pts = predicted_key_pts.numpy()


            # undo normalization of keypoints  
            #predicted_key_pts = predicted_key_pts*50+100

            ground_truth_pts = gt_pts[i]  
            if (torch.cuda.is_available()):
                ground_truth_pts = ground_truth_pts.cpu()

            ground_truth_pts = ground_truth_pts.numpy()
            #ground_truth_pts = ground_truth_pts*50+100

            RMSE = mean_squared_error(ground_truth_pts, predicted_key_pts, squared=False)
            #print(name[i], RMSE)
            print('{} RMSE: {}'.format(name[i], RMSE))
            RMSE_total += RMSE
            total_num += 1
        RMSE_avg = RMSE_total / total_num  
        print('RMSE:', RMSE_avg)

    RMSE_avg = RMSE_total / total_num   
    print('RMSE:', RMSE_avg)


            

        



        
        
        
        




