import json
import os
import numpy as np


if __name__ == "__main__":
    anns_dir = "data/wibam/annotations/hand_labels/"
    json_files = [pos_json for pos_json in os.listdir(anns_dir) if pos_json.endswith(".json")]
    
    visibilities = []
    elevations = []
    distances = []
    cam_loc = {}
    cam_loc[0] = np.array([-0.4,33.3,3.3])
    cam_loc[1] = np.array([27.8,39.7,3.5])
    cam_loc[2] = np.array([34.7,10.9,3.8])
    cam_loc[3] = np.array([6.4,5.1,3.4])

    visible_objects = 0

    for jfile in json_files:
        with open(anns_dir + jfile, "r") as file:
            saved_dict = json.load(file)
            anns = saved_dict["annotations"]
            img_info = saved_dict["ann_info"]
            sync_offsets = img_info["sync_offsets"]
            file_num = int(jfile.split(".")[0])
            for i in range(len(sync_offsets)):
                img_num = file_num + sync_offsets[i]
                img_anns = {}
                for j in range(len(anns)):
                    obj = anns[str(j)]
                    if obj["visibility"][i] > 0:
                        
                        cam_coords = cam_loc[i]
                        obj_coords = np.array([obj["x"], obj["y"]])
                        distance = np.linalg.norm(cam_coords[:2]-obj_coords)
                        elevation = np.arctan(cam_coords[2]/distance) * 180/np.pi

                        # Record stats
                        distances.append(distance)
                        elevations.append(elevation)
                        visibilities.append(obj["visibility"][i])
                        visible_objects += 1
                        
                        img_anns[j] = obj
                
                save_dir = anns_dir + "{}/".format(i)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file = save_dir + str(img_num) + ".json"
                with open(save_file, "w") as write_file:
                    json.dump(img_anns, write_file, sort_keys=True, indent=4)

    distances = np.array(distances)
    elevations = np.array(elevations)
    visibilities = np.array(visibilities)

    stat_file = anns_dir + "stats.csv"
    with open(stat_file, "w") as file:
        np.savetxt(file, [distances, elevations, visibilities], delimiter=",")

    print(visible_objects)
