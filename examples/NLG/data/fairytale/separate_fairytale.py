import sys
import os
import json
import random
import math

def write_into_file(data, writer):
    for (prev, curr) in data:
        x = {}
        x['context'] = prev
        x['completion'] = curr
        writer.write(json.dumps(x) + '\n')
    return

with open('ft_all_clean.txt', 'r', encoding='utf8') as reader, \
     open('ft_train_formatted.jsonl', 'w', encoding='utf8') as train, \
     open('ft_valid_formatted.jsonl', 'w', encoding='utf8') as val, \
     open('ft_test_formatted.jsonl', 'w', encoding='utf8') as test:
    file = reader.read()
    sentences = [line.strip(' ') for line in file.split('\n')]
    sentences = [line for line in sentences if line != '']
    data = []
    for i in range(1, len(sentences)):
        data.append((sentences[i-1], sentences[i]))

    random.shuffle(data)
    num_data = len(data)
    num_train = math.ceil(num_data * .9)
    num_val = (num_data - num_train) // 2
    num_test = num_data - num_train - num_val

    write_into_file(data[:num_train], train)
    write_into_file(data[num_train:num_train + num_val], val)
    write_into_file(data[num_train + num_val:], test)

