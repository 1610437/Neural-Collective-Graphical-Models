import torch
import tqdm

class Data_loader():
    def __init__(self, population_data, location, time_size, location_size, device):
        self.input_list = []
        self.population_list = []
        self.device = device
        
        for t in tqdm.trange(time_size):
            input_list_tmp = []
            for l in range(location_size):
                input_tmp = []
                for ll in range(location_size):
                    input_tmp.append([t / float(time_size) - 0.5, location[l][0], location[l][1], location[ll][0] - location[l][0], location[ll][1] - location[l][1]])
                input_list_tmp.append(input_tmp)
            self.input_list.append(torch.tensor(input_list_tmp, dtype=torch.double))
            self.population_list.append(torch.tensor(population_data[t], dtype=torch.double))
    
    def get_t_input(self, t):
        return self.input_list[t].to(self.device), self.population_list[t].to(self.device), self.population_list[t + 1].to(self.device)
