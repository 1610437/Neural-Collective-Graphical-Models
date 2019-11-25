import csv
import numpy as np
from pathlib import Path
import torch

with open(str("loc.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
	reader = csv.reader(csvfile)
	loc = [[int(col) for col in row] for row in reader]
loc=torch.tensor(loc)
z=torch.zeros([28,28],dtype=torch.double)
print(loc)

#print(adj)
for l in range(28):
	for ll in range(28):
		if max(abs(loc[ll,0]-loc[l,0]),abs(loc[ll,1]-loc[l,1]))<=3:
			z[l,ll]=torch.tensor(max(abs(loc[ll,0]-loc[l,0]),abs(loc[ll,1]-loc[l,1])))+1

for l in range(28):
	z[l,l]=1
print(z)

f = open('z333.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
for i in range(28):
     writer.writerow(z[i,:].detach().numpy())
f.close()                          

