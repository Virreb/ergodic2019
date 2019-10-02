import os
import json

path = 'data_raw/Training_dataset/Percentages'
json_files = os.listdir(path)

total_water = 0
total_building = 0
total_road = 0
for file_name in json_files:
    with open(path + '/' + file_name, 'r') as f:
        perc = json.load(f)
        total_water += perc['water']
        total_building += perc['building']
        total_road += perc['road']


total_water = total_water / len(json_files)
total_building = total_building / len(json_files)
total_road = total_road / len(json_files)
total_void = 100-total_road-total_building-total_water

print(f'void: {1}')
print(f'road: {total_void/total_road}')
print(f'water: {total_void/total_water}')
print(f'building: {total_void/total_building}')


