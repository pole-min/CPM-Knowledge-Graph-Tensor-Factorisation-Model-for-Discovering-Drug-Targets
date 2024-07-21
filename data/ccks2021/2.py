tri=[]
with open('train.txt','r') as f:
    for line in f:
        tri.append(line.strip())
with open('test.txt','r') as f:
    for line in f:
        tri.append(line.strip())
print(len(tri))
train=[]
for line in tri:
    head, rel, tail = line.split('\t')
    if rel not in ['interacts_with']:
        train.append(line)
print(len(train))
with open('../ccks2021_/train_.txt','w') as f:
   for line in train:
       f.write(line+'\n')