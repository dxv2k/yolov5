import json
import yaml
import train 
import shutil
from pathlib import Path
from clearml import Task





# Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')

import shutil
def unzip_training_dataset(filepath:str, extract_dir: str) -> Path: 
    '''
    '''
    shutil.unpack_archive(filepath, extract_dir)
    return Path(extract_dir)  


def create_yaml_schema() -> str: 
    extract_dir = unzip_training_dataset(
        "/media/dxv2k/Work & Data/viAct/test_fiftyone/yolov5/temp_dataset/label_studio_export_yolo.zip", 
        "/media/dxv2k/Work & Data/viAct/test_fiftyone/yolov5/temp_dataset/"
    )
    
    files = sorted(list(
        Path(extract_dir).iterdir()
    ))
    dataset_path = extract_dir
    images_path = None
    labels_path = None
    classes_path = None

    for f in files: 
        if "images" in str(f.absolute()): 
            images_path = f.absolute() 
        
        if "labels" in str(f.absolute()): 
            labels_path = f.absolute() 

        if "notes.json" in str(f.absolute()): 
            classes_path = f.absolute() 


    with open(classes_path.absolute(),"r") as f: 
        classes_json = json.load(f)

    yaml_classes_name = []
    for val in classes_json['categories']: 
        new_dict = { 
            val.get('id'): val.get('name')
        }
        yaml_classes_name.append(new_dict)
        # break
    # print(yaml_classes_name)
    yolov5_yaml_schema = { 
        "path": str(dataset_path.absolute()), 
        "train": str(images_path.absolute()), 
        "val": "", 
        "test": "",
        "names": yaml_classes_name, # classes [idx: cls_name]
        "download": "" # Optional 
    }



    # pprint(yolov5_yaml_schema)
    yaml_path = extract_dir / "train.yaml" 
    with open(yaml_path,"w+") as f: 
        yaml.dump(yolov5_yaml_schema, f)
        
    return yaml_path 



def execute(): 
    yaml_path = create_yaml_schema()
    print("Complete prepared dataset, continue to training the model...")
    train.run(
        data=yaml_path, 
        imgsz=640,  
        weights='yolov5s.pt'
    )

def run_me_remotely(some_argument):
    print(some_argument)


# execute()

task = Task.init(project_name="YOLOv5", task_name="test_remote_execute")
train_task = task.create_function_task(
    func=execute,  # type: Callable
    func_name='train_yolov5',  # type:Optional[str]
    task_name='test remote train yolov5',  # type:Optional[str]
    # everything below will be passed directly to our function as arguments
    # some_argument=None
)
