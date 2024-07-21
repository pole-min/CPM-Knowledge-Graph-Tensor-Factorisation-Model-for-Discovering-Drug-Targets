with open('entity2text.txt','r') as f:
    entity2text = f.readlines()
entities=[]
with open('entities.txt','r') as f:
    for line in f:
        entities.append(line.strip())
l=[]
ll=[]
for line in entity2text:
    try :
        ent,text = line.split('\t')
    except:
        print(line)
    if ent not in l and ent in entities:
        l.append(ent)
        ll.append(line)
print(len(l))
l3=set(entities).difference(set(l))
print(len(l3))
for ent in l3:
    ll.append(ent+'\t'+ent+'\n')
print(len(ll))
with open('entity2text.txt','w') as f:
    f.writelines(ll)

