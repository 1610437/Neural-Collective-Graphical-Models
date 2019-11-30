from pathlib import Path
import tqdm
from collections import OrderedDict
import numpy as np

import torch
import torch.optim as optim

import tensorboardX

import model
import datas
import dataloader
import csv

stay_ratio=0.8

if __name__ == "__main__":
    #Select data which you use
    is_sample = 2
    #Hyper parameter for objective function
    lam=10.0
    #Dimention of input layer and hidden layer
    input_layer=5
    hidden_layer=40
    
    if is_sample==1:
        population_data, location, adj_table, z_table = datas.read_sample()
    elif is_sample==2:
        population_data, location_table, adj_table, z_table = datas.read_chofu()
        location = [[row[0] / 6 - 0.5, row[1] / 6 - 0.5] for row in location_table]
    else:
        population_data, adj_table, location_table, neighbor_table = datas.read_data(Path("datas/chohu"), "chohu_01.csv", False)
        location = [[row[0] / 11 - 0.5, row[1] / 14 - 0.5] for row in location_table]


    time_size = torch.tensor(population_data).shape[0]
    location_size = torch.tensor(population_data).shape[1]
    z_tensor=torch.zeros(time_size-1,location_size,location_size,dtype=torch.double)
    for l in tqdm.trange(time_size-1):
            for ll in range(location_size):
                #for lll in range(location_size):
                #z_tensor[l,ll,:]=torch.tensor([0.8],dtype=torch.double).pow(z_table[ll,:])*torch.tensor(population_data[l]).sum()
                #z_tensor[l,ll,:]=z_table[ll,:]*torch.tensor(population_data[l]).sum()/adj_table[ll,:].sum()/5#weight1
                #weight2,3傾斜なしと傾斜あり
                z_tensor[l,ll,:]=adj_table[ll,]*torch.tensor(population_data[l])[ll]*0.2
                z_tensor[l,ll,ll]=torch.tensor(population_data[l])[ll]*adj_table[ll,ll]*0.8
                #z_tensor[l,ll,:]=adj_table[ll,:]*model.digit#noweight
                    #if z_table[ll,lll]>0:
                        #z_tensor[l,ll,lll]=(-((1/z_table[ll,lll]))).exp()*torch.tensor(population_data[l])[ll]
    #print(z_tensor[0,0,:])
    f = open('outputcsv/zinit11_26.csv', 'w')
    writer = csv.writer(f, lineterminator='\n')
    for l in range(time_size-1):
        for ll in range(location_size):
           writer.writerow(z_tensor[l,ll,:].detach().numpy())
    f.close()
    
    #Use cuda
    use_cuda = True
    available_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if (use_cuda and available_cuda) else 'cpu')
    print(device)

    #Set default type of tensor
    torch.set_default_dtype(torch.double)
    #torch.set_grad_enabled(True)
    #torch.autograd.set_detect_anomaly(True)

    #Use tensorboardX
    board = tensorboardX.SummaryWriter()

    #Instantinate model
    mod = model.NCGM(input_layer, hidden_layer,z_tensor)
    mod.to(device)

    #Instantinate objective function
    objective = model.NCGM_objective(location_size,adj_table)

    #Instantinate optimizer
    #optimizer = optim.SGD(mod.parameters(), lr=0.5)
    optimizer = optim.Adam(mod.parameters())

    #Instantinate dataloader
    data_loader = dataloader.Data_loader(population_data, location, time_size, location_size, device)
    
    #Training
    mod.train()
    itr = tqdm.trange(2000)
    #itr = tqdm.trange(1)
    losses = []
    ave_loss = 0.0
    for i in itr:
        for t in range(time_size - 1):
        #for t in range(1):
            input_data, yt, yt1 = data_loader.get_t_input(t)
            theta = mod(input_data)
            loss = objective(theta, mod.Z[t], yt, yt1, lam)
            #print(loss)
            losses.append(loss.item())
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #itr.set_postfix(ordered_dict=OrderedDict(loss=loss.item(), b_grad=mod.fc2.bias.grad))
            #itr.set_postfix(ordered_dict=OrderedDict(loss=loss.item()))

            board.add_scalar("loss", loss.item(), i * (time_size - 1) + t)
            ave_loss = ave_loss + loss.item()
            
            
        itr.set_postfix(ordered_dict=OrderedDict(loss=ave_loss/(time_size-1),))    
        board.add_text("Z", str(mod.Z), i)
        board.add_scalar("ave_loss", ave_loss / (time_size - 1), i)
        ave_loss = 0.0

        #with open("output/{0:05}.txt".format(i), 'wt') as f:
          #  f.write(str(mod.Z.data.numpy()))
    
    #tensorboard用の値のjsonファイルへの保存[ポイント6]
    board.export_scalars_to_json("./all_scalars.json")
    board.add_text("progress", "finish", 0)
    #SummaryWriterのclose[ポイント7]
    board.close()
    
    f = open('outputcsv/z11_26N11.csv', 'w')
    writer = csv.writer(f, lineterminator='\n')
    for l in range(time_size-1):
    #for l in range(1):
        for ll in range(location_size):
            writer.writerow(mod.Z[l,ll,:].detach().numpy()*model.digit)
    f.close()
    
    f = open('outputcsv/theta11_26N11.csv', 'w')
    writer = csv.writer(f, lineterminator='\n')
    for l in range(location_size):
        writer.writerow(theta[l,:].detach().numpy())
    f.close()
    