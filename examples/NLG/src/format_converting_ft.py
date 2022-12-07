import sys
import os
import numpy as np
import pandas as pd
import json
import random
import math
from spacy.lang.en import English
from tqdm import tqdm

with open(os.path.join(sys.argv[1], 'train_formatted.jsonl'), 'w', encoding='utf8') as train, \
     open(os.path.join(sys.argv[1], 'test_formatted.jsonl'), 'w', encoding='utf8') as test, \
     open(os.path.join(sys.argv[1], 'valid_formatted.jsonl'), 'w', encoding='utf8') as valid:
    #  open(os.path.join(sys.argv[1], 'stories_splitted.jsonl'), 'w', encoding='utf8') as f_all:
    
    data = pd.read_parquet(os.path.join(sys.argv[1], sys.argv[2]), engine='pyarrow')
    print(data['url'][0])
    print(data['text'][0])

    num_stories = len(data['text'])
    data_all = []

    nlp = English()
    nlp.add_pipe('sentencizer')

    data_bar = tqdm(range(num_stories), desc='data')
    for (idx, story) in enumerate(data['text']):
        if len(story) >= 1000000:
            print('story %d is too long, len = %d, skipped' % (idx + 1, len(story)))
            data_bar.update()
            continue

        story = story.replace('\u00a0', ' ')
        doc = nlp(story)
        sentences = []
        for utterance_span in doc.sents:
            utterance = ' '.join([token.text for token in utterance_span])
            sentences.extend([utter.strip(' ') for utter in utterance.split('\n') if utter.strip(' ') != ''])
        
        y = {}
        y['story_id'] = idx + 1
        y['story'] = sentences
        # f_all.write(json.dumps(y) + '\n')

        x = {}
        for i in range(2, len(sentences)):
            x['context'] = 'previous: ' + sentences[i - 2] + ' ' + sentences[i - 1]
            x['completion'] = sentences[i]
            data_all.append(x.copy())
        
        data_bar.update()
    data_bar.close()
    
    random.shuffle(data_all)
    num_data = len(data_all)
    data_part = data_all[:num_data // 10]
    num_data = len(data_part)
    num_train = math.ceil(num_data * .99)
    num_val = (num_data - num_train) // 2
    num_test = num_data - num_train - num_val

    train_bar = tqdm(range(num_train), desc = 'train')
    for x in data_part[ : num_train]:
        train.write(json.dumps(x) + '\n')
        train_bar.update()
    train_bar.close()
    
    val_bar = tqdm(range(num_val), desc = 'valid')
    for x in data_part[num_train : num_train + num_val]:
        valid.write(json.dumps(x) + '\n')
        val_bar.update()
    val_bar.close()
    
    test_bar = tqdm(range(num_test), desc = 'test')
    for x in data_part[num_train + num_val : ]:
        test.write(json.dumps(x) + '\n')
        test_bar.update()
    test_bar.close()

