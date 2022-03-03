import multiprocessing as mp
import random
import string
import time

pet_maps = {
        "adam": {"pet_name": "max"},
        "steve": {"pet_name": "sylvester"},
        "michelle": {"pet_name": "fuzzy"},
        "frank": {"pet_name": "pete"},
        "will": {"pet_name": "cat"},
        "natasha": {"pet_name": "tweety"},
        "samantha": {"pet_name": "bob"},
        "peter": {"pet_name": "garfield"},
        "susan": {"pet_name": "zazu"},
        "josh": {"pet_name": "tom"},
    }

pet_owners = pet_maps.keys()

output = mp.Queue()

def get_pet_name(data, output):
    time.sleep(5)
    print('adding to queue', data)
    response = 'pet name: {}'.format(data)
    output.put(response)

processes = [mp.Process(target=get_pet_name, args=(pet_maps[name]['pet_name'], output)) for name in pet_owners]

for p in processes:
    p.start()

#for p in processes:
#    p.join()

print('consuming from queue:')
#results = [output.get() for p in processes]

for p in processes :
    
    print(output.get())

#print(results)
