def normalize(name):
    a = []
    for i in name:
        if ord(i) > 91:
            i = chr(ord(i) - 32)
            a.append(i)
        else:
            a.append(i)
    return str(''.join(a))

#def normalize1(name):
    return name.title()

def normalize1(name):
    return name[0].upper() + name[1:].lower()
        
L1 = ['adam', 'LISA', 'barT']
L2 = list(map(normalize, L1))
L3 = list(map(normalize1, L1))
print(L2)
print(L3)
         

