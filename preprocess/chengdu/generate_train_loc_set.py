import os
import pickle

input_dir = '/data/jinyang/map_match/chengdu/final_trips'
namelist = os.listdir(input_dir)
chengdu_roads = set([])

for name in namelist:
    fr = open(os.path.join(input_dir,name),'rb')
    trips = pickle.load(fr)
    for key in trips.keys():
        paths = trips[key]
        for i in range(0,len(paths)):
            roadid = paths[i][2]
            chengdu_roads.add(roadid)
    print(name)

print(chengdu_roads)
fw = open('/data/wuning/RN-GNN/chengdu/locList','wb')
pickle.dump(chengdu_roads,fw)

# fw = open('roads','rb')
# road_use = pickle.load(fw)
# print(len(road_use))
