import numpy as np
import torch
from flask import Flask, jsonify, request

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pydantic import BaseModel
from statistics import mode
import logging


app = Flask(__name__)

# class EntityIn(BaseModel):
#     texts : list



# class EntityOut(BaseModel):
#     label : str


@app.route("/",methods=['GET'])
def test():
    return jsonify({"Hello" : "World"})

def get_model():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('prakhars/instagram_caption_classifier')
    return tokenizer, model 

label_dict= {
14: 'Travel',
 0: 'Art and Culture',
 9: 'Music',
 4: 'Entertainment',
 10: 'Parenting',
 8: 'Health and Fitness',
 13: 'Sports',
 6: 'Food',
 2: 'Beauty, Fashion and Lifestyle',
 1: 'Automotive',
 11: 'Pets',
 5: 'Finance',
 12: 'Photography',
 3: 'DIY',
 7: 'Gaming'
 }




tokenizer, model = get_model()




@app.route('/predict/',methods=['POST'])

def read_root():
    
    try:
        texts = request.json['texts']
        encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**encodings)
        predictions = outputs.logits.argmax(-1)
        out_codes = predictions.tolist()
        out_labels = {'labels':[]}
        most_common_label = {}
        for code in out_codes:
            out_labels['labels'].append(label_dict[code])
    except Exception as e:
        logging.error("Exception occured", exc_info=True)

    try :
        most_common_label['label'] = mode(out_labels['labels'])
    except:
        most_common_label['label'] = out_labels['labels'][0]

    

    return jsonify(most_common_label)

if __name__=='__main__':
    app.run(port=8000)
    

