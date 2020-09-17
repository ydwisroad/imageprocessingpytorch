import os
import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

path = '../../../../data/custimzedCocoData/val'

files = []

#list all json files and append to array
for file in os.listdir(path):
    if file[-5:] == '.json':
        files.append(file)

#print(json.load(open(path+os.sep +files[0])))

via_region_data = {}

for file in files:
    jsonPath = path+os.sep+file
    print('json Path ', jsonPath)
    one_json = json.load(open(jsonPath))
    #print('loaded json  ', one_json)

    one_image = {}
    one_image["filename"] = file.split('.')[0] + '.jpg'
    shape = one_json["shapes"]

    regions = {}
    for i in range(len(shape)):
        points = np.array(shape[i]['points'])

        all_points_x = list(points[:,0])
        all_points_y = list(points[:,1])

        regions[str(i)] = {}
        regions[str(i)]['region_attributes'] = {}
        regions[str(i)]['shape_attributes'] = {}

        regions[str(i)]['shape_attributes']['all_points_x'] = all_points_x
        regions[str(i)]['shape_attributes']['all_points_y'] = all_points_y

        regions[str(i)]['shape_attributes']['name'] = shape[i]['label']

    one_image['regions'] = regions
    one_image['size'] = 0

    via_region_data[file] = one_image

print(via_region_data)

with open(path+os.sep + 'viaregiondata.json', 'w') as f:
    json.dump(via_region_data, f, sort_keys=False, ensure_ascii=True,  cls=NpEncoder)

