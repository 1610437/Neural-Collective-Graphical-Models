import torch
import torch.nn as nn

digit=1000

class NCGM(nn.Module):
    def __init__(self, input_size, hidden_size,z):
        super(NCGM, self).__init__()

        #Dimention of input layer
        self.input_size = input_size
        #Dimention of hidden layer
        self.hidden_size = hidden_size
        #People flow parameter
        self.Z = nn.Parameter(z/digit)
        
        #Network for solving θ
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Softmax(1)

    def forward(self, input):
        out = self.fc1(input).tanh()
        out = self.fc2(out)
        out = self.softmax(out)
        theta = out.squeeze()
        return theta
    
class NCGM_objective(nn.Module):
    def __init__(self, location_size,adj_table):
        super(NCGM_objective, self).__init__()

        #Number of areas
        self.L = location_size
        #Neighbour matrix
        self.adj_table=adj_table
        
        #Loss functions
        self.mse_loss_t = nn.MSELoss(reduction='sum')
        self.mse_loss_t1 = nn.MSELoss(reduction='sum')
    
    def forward(self, theta, Z, yt, yt1, lam):
        
        #Set lower limit
        theta_log = theta.clamp(min=3.7835e-10).log()
        Z_log = Z.clamp(min=3.7835e-10).log()
        #theta_log = theta.clamp(min=1.0e-4).log()
        #Z_log = Z.clamp(min=1.0e-4).log()
        #Figure out L' (while obj_arr is a square matrix and obj_L is scalar)
        obj_arr = Z.mul(theta_log.add(1).add(-1, Z_log))
        obj_L=0
        Z1=torch.zeros(self.L,dtype=torch.double)
        Z2=torch.zeros(self.L,dtype=torch.double)
        
        #Sum up every elements in obj_arr under adj_table
        #Sum up every lows in Z to make transposed matrix under adj_table
        obj_arr=obj_arr*self.adj_table
        Z=Z*self.adj_table
        obj_L=obj_arr.sum()
        Z1=Z.sum(axis=1)
        Z2=Z.sum(axis=0)

        #Limit for people from other areas
        et = self.mse_loss_t(yt/digit, Z1)
        #Limit for people to other areas
        et1 = self.mse_loss_t1(yt1/digit, Z2)
        
        #Figure out G
        G = obj_L.add(-1*lam, et.add(1, et1))
        
        #Multiple -1 to G
        return G.neg()