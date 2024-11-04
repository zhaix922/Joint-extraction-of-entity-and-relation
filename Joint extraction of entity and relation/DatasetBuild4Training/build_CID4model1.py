# rebuild CID dataset for training BIO_Rel_Model4single_sentence

import json
import nltk
import argparse


def parse_option():
    parser = argparse.ArgumentParser()
    # r"/home/yezaisen/BM_GBD/Dataset/original_CID_dataset_test.txt"
    parser.add_argument('--original_dataset_file',type = str,default=r"/data10/yezaisen/program/nl2sql_datagenerate/tools/临时文件/original_file.txt")
    parser.add_argument('--dataset_save_path',type = str,default = r"/data10/yezaisen/program/nl2sql_datagenerate/tools/临时文件/new_file.jsonl")
    args = parser.parse_args()
    return args


def get_new_position(original_text,sentences_list,position):
    text = original_text
    assert len(text) > position[1],"DEBUG ERROR"
    entity_start_pos = position[0]
    entity_tail_pos = position[1]
    for sent_index,sentence in enumerate(sentences_list):
        sentence_start_pos = text.find(sentence)
        if sentence_start_pos > 0:
            entity_start_pos -= sentence_start_pos
            entity_tail_pos -= sentence_start_pos
            sentence_start_pos = 0
        if len(sentence) < entity_start_pos:
            text = text[sentence_start_pos + len(sentence):]
            entity_start_pos -= len(sentence)
            entity_tail_pos -= len(sentence)
        else:
            return (entity_start_pos,entity_tail_pos),sent_index
    raise 
    

def CID_dataset_load(file:str):
    dataset = []
    with open(file,'r',encoding='utf-8') as f:
        single_data = None
        data_parse_step = 0
        i = 0
        for single_line in f:
            single_line = single_line.strip(" \n\t")
            if data_parse_step == 0:
                data_id,title_flag,title_text = single_line.split('|')
                single_data = {"Sentences":[],"Title":title_text,"Entity":[],"Relation":[],"Text":None}
                data_parse_step += 1
            elif data_parse_step == 1:
                data_id,text_flag,text = single_line.split('|')
                sentences_list = nltk.tokenize.sent_tokenize(text)
                title_text = single_data["Title"]
                Sentences = [title_text] + sentences_list
                single_data["Sentences"] = Sentences
                single_data["Text"] = title_text + ' ' + text
                data_parse_step += 1
            elif data_parse_step == 2 and single_line != "":
                split_part = single_line.split('\t')
                if len(split_part) > 4:
                    data_id,start_pos,tail_pos,word,entity_type,entity_code = single_line.split('\t')
                    entity_index = len(single_data["Entity"])
                    entity_position = (int(start_pos),int(tail_pos))
                    (new_start_pos,new_tail_pos),sent_id = get_new_position(single_data["Text"],single_data["Sentences"],entity_position)
                    single_data["Entity"].append({
                            "Pos":(new_start_pos,new_tail_pos),
                            "Word":word,
                            "Entity_id":entity_index,
                            "Entity_code":entity_code,
                            "Type":entity_type,
                            "Sent_id":sent_id,
                        })
                else:
                    if single_line == "435349	CID	D013390	D005207":
                        pass
                    data_id,relation_type,entity_code1,entity_code2 = single_line.split('\t')
                    entity_head_index = -1
                    entity_tail_index = -1
                    head_source_sent_index = -1
                    tail_source_sent_index = -1
                    save_flag = False
                    # distance = (sent_distance,position_distance)
                    distance = (1e9,1e9)
                    for i,entity1 in enumerate(single_data["Entity"]):
                        if entity1["Entity_code"] == entity_code1:
                            same_s_entity_index = None
                            same_s_pos_dis = 1e9
                            for j,entity2 in enumerate(single_data["Entity"]):
                                if entity2["Entity_code"] == entity_code2:
                                    sent_dis = abs(entity1["Sent_id"] - entity2["Sent_id"])
                                    pos_dis = min(
                                        abs(entity1["Pos"][0] - entity2["Pos"][1]),
                                        abs(entity1["Pos"][1] - entity2["Pos"][0]),
                                    )
                                    temp_dis = (sent_dis,pos_dis)
                                    if temp_dis < distance:
                                        distance = temp_dis
                                        entity_head_index = i
                                        entity_tail_index = j
                                        head_source_sent_index = entity1["Sent_id"]
                                        tail_source_sent_index = entity2["Sent_id"]
                                    if temp_dis[0] == 0:
                                        if pos_dis < same_s_pos_dis:
                                            
                                            same_s_pos_dis = pos_dis
                                            same_s_entity_index = j
                                        
                            if temp_dis[0] == 0:
                                save_flag = True
                                source = single_data["Sentences"][head_source_sent_index]
                                single_data["Relation"].append({
                                    "Relation_type":relation_type,
                                    "Head_entity":i,
                                    "Tail_entity":same_s_entity_index,
                                    "Source":source
                                })
                    if save_flag:
                        continue
                    if head_source_sent_index == tail_source_sent_index:
                        source = single_data["Sentences"][head_source_sent_index]
                    else:
                        source = single_data["Sentences"][head_source_sent_index] + "|" + single_data["Sentences"][tail_source_sent_index]
                        
                    single_data["Relation"].append({
                        "Relation_type":relation_type,
                        "Head_entity":entity_head_index,
                        "Tail_entity":entity_tail_index,
                        "Source":source
                    })
            elif data_parse_step == 2 and single_line == "":
                data_parse_step = 0
                dataset.append(single_data)
            else:
                raise KeyError("NEED DEBUG")
    return dataset

def main():
    args = parse_option()
    original_CID_dataset = CID_dataset_load(args.original_dataset_file)
    with open(args.dataset_save_path,'w',encoding='utf-8') as f:
        for single_data in original_CID_dataset:
            sds = json.dumps(single_data,ensure_ascii=False)
            f.writelines(sds + '\n')

if __name__=="__main__":
    main()

