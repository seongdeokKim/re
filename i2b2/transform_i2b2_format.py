import os
import re

def read_sent_info(id):

    sent_list = []
    with open('./txt/' + str(id) + ".txt", "r", encoding='utf-8') as fr:
        for line in fr:
            sent_list.append(
                line.strip().split()
            )

    return sent_list


def read_rel_info(id):

    rel_info_list = []

    pattern_a = r'c="(.+)" (\d+):(\d+) (\d+):(\d+)'
    pattern_b = r'r="(.+)"'

    with open('./rel/' + str(id) + ".rel", "r", encoding='utf-8') as fr:
        for line in fr:
            field_line = line.strip().split('||')

            left_entity_field = field_line[0]
            sent_id = re.search(pattern_a, left_entity_field).group(2)
            left_entity = re.search(pattern_a, left_entity_field).group(1)
            left_start_pos = re.search(pattern_a, left_entity_field).group(3)
            left_end_pos = re.search(pattern_a, left_entity_field).group(5)

            relation_field = field_line[1]
            relation = re.search(pattern_b, relation_field).group(1)

            right_entity_field = field_line[2]
            right_entity = re.search(pattern_a, right_entity_field).group(1)
            right_start_pos = re.search(pattern_a, right_entity_field).group(3)
            right_end_pos = re.search(pattern_a, right_entity_field).group(5)

            rel_info_per_line = {
                "sent_id": sent_id,
                "left_entity": left_entity,
                "left_start_pos": left_start_pos,
                "left_end_pos": left_end_pos,
                "relation": relation,
                "right_entity": right_entity,
                "right_start_pos": right_start_pos,
                "right_end_pos": right_end_pos,
            }

            rel_info_list.append(rel_info_per_line)

    return rel_info_list


def read_con_info(id):

    con_info_list = []

    pattern_a = r'c="(.+)" (\d+):(\d+) (\d+):(\d+)'
    pattern_b = r't="(.+)"'

    with open('./concept/' + str(id) + ".con", "r", encoding='utf-8') as fr:
        for line in fr:
            field_line = line.strip().split('||')

            entity_field = field_line[0]
            entity = re.search(pattern_a, entity_field).group(1)
            sent_id = re.search(pattern_a, entity_field).group(2)
            start_pos = re.search(pattern_a, entity_field).group(3)
            end_pos = re.search(pattern_a, entity_field).group(5)

            concept_field = field_line[1]
            concept = re.search(pattern_b, concept_field).group(1)

            con_info_per_line = {
                "entity": entity,
                "sent_id": sent_id,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "concept": concept
            }

            con_info_list.append(con_info_per_line)

    return con_info_list


if __name__ == '__main__':

    # Read all if id list
    id_list = []
    files = os.listdir('./rel')
    for file in files:
        #print(file.strip().replace('.rel', ''))
        id_list.append(file.strip().replace('.rel', ''))

    # Map each id with info. from "id.txt" and "id.rel" and "id.con"
    id_dict = {}
    for id in id_list:
        sent_list = read_sent_info(id)
        con_info_list = read_con_info(id)
        rel_info_list = read_rel_info(id)

        id_dict[id.strip()] = (sent_list, con_info_list, rel_info_list)

    with open('transformed_i2b2_input.txt', 'w', encoding='utf-8') as fw:
        cnt = 0
        for id, (sent_list, con_info_list, rel_info_list) in id_dict.items():

            entity_dict_list = []
            for con_info_per_line in con_info_list:
                entity = con_info_per_line.get("entity")
                sent_id = con_info_per_line.get("sent_id")
                start_pos = con_info_per_line.get("start_pos")
                end_pos = con_info_per_line.get("end_pos")
                concept = con_info_per_line.get("concept")

                entity_dict = {
                    "concept": concept,
                    "entity": entity,
                    "sent_id": sent_id,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                }
                entity_dict_list.append(entity_dict)

            # Write the whole features of each sample for relation extraction  
            for rel_info_per_line in rel_info_list:

                sent_id = rel_info_per_line.get("sent_id")
                relation = rel_info_per_line.get("relation")
                
                left_entity = rel_info_per_line.get("left_entity")
                left_start_pos = rel_info_per_line.get("left_start_pos")
                left_end_pos = rel_info_per_line.get("left_end_pos")

                right_entity = rel_info_per_line.get("right_entity")
                right_start_pos = rel_info_per_line.get("right_start_pos")
                right_end_pos = rel_info_per_line.get("right_end_pos")

                # Get the concept of left entity
                # If info. of "*.con" file is same with info. of "*.rel" file, get the concept according to the entity
                left_concept = None
                for entity_dict in entity_dict_list:
                    if entity_dict.get("entity") != left_entity: continue
                    if entity_dict.get("sent_id") != sent_id: continue
                    if entity_dict.get("start_pos") != left_start_pos: continue
                    if entity_dict.get("end_pos") != left_end_pos: continue
                    
                    left_concept = entity_dict.get("concept")

                # Get the concept of right entity
                # If info. of "*.con" file is same with info. of "*.rel" file, get the concept according to the entity
                right_concept = None
                for entity_dict in entity_dict_list:
                    if entity_dict.get("entity") != right_entity: continue
                    if entity_dict.get("sent_id") != sent_id: continue
                    if entity_dict.get("start_pos") != right_start_pos: continue
                    if entity_dict.get("end_pos") != right_end_pos: continue
                    
                    right_concept = entity_dict.get("concept")

                sent = sent_list[int(sent_id) - 1]

                # Get the span info. of left entity
                # i2b2 dataset provides us the position of sentence unit (left_start_pos, left_end_pos)
                # So need to convert position of token unit to position of character unit
                left_entity_span = None
                if int(left_start_pos) != 0:
                    start_pos = len(' '.join(sent[: int(left_start_pos)])) + 1
                    end_pos = start_pos + len(' '.join(sent[int(left_start_pos): int(left_end_pos) + 1]))
                else:
                    start_pos = 0
                    end_pos = start_pos + len(' '.join(sent[: int(left_end_pos) + 1]))

                if left_entity == ' '.join(sent)[start_pos: end_pos].lower():
                    left_entity_span = str(start_pos) + '_' + str(end_pos)

                # Get the span info. of right entity
                # i2b2 dataset provides us the position of sentence unit (right_start_pos, right_end_pos)
                # So need to convert position of token unit to position of character unit
                right_entity_span = None
                if int(right_start_pos) != 0:
                    start_pos = len(' '.join(sent[: int(right_start_pos)])) + 1
                    end_pos = start_pos + len(' '.join(sent[int(right_start_pos): int(right_end_pos) + 1]))
                else:
                    start_pos = 0
                    end_pos = start_pos + len(' '.join(sent[: int(right_end_pos) + 1]))
                    
                if right_entity == ' '.join(sent)[start_pos: end_pos].lower():
                    right_entity_span = str(start_pos) + '_' + str(end_pos)

                # Write all features per example                  
                if left_concept is not None and right_concept is not None:
                    if left_entity_span is not None and right_entity_span is not None:
                        #print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                        #    relation,
                        #    left_entity, left_concept, left_entity_span,
                        #    right_entity, right_concept, right_entity_span,
                        #    ' '.join(sent), id
                        #))
                        fw.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                            relation,
                            left_entity, left_concept, left_entity_span,
                            right_entity, right_concept, right_entity_span,
                            ' '.join(sent), id
                        ))
