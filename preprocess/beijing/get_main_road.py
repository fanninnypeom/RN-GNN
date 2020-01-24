import json
road_types = ["motorway_link", "trunk_link", "primary_link", "motorway", "trunk", "primary", "secondary", "secondary_link"]#, "secondary"]
roadNet = json.load(open("/data/wuning/map-matching/osmextract-node.json","r"))
fo = open("/data/wuning/RN-GNN/beijing/main_road.js", "w")
roads = []
for item in roadNet["features"]:
  if "highway" in item["properties"] and not (item["properties"]["highway"] in road_types):
    roads.append(item["geometry"]["coordinates"])  
fo.write(str(roads))
    
