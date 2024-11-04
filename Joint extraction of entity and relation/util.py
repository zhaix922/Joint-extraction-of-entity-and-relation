
import json


def json_file_load(file:str):
    
    if file.endswith("json"):
        with open(file,'r',encoding='utf-8') as f:
            file_content = json.load(f)
        return file_content
    elif file.endswith("jsonl"):
        dataset = []
        with open(file,'r',encoding='utf-8') as f:
            for single_line in f:
                single_data = json.loads(single_line)
                dataset.append(single_data)
        return dataset


