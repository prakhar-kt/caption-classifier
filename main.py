import numpy as np
import torch
from fastapi import FastAPI, Request
import uvicorn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pydantic import BaseModel


app = FastAPI()

class EntityIn(BaseModel):
    texts : list



class EntityOut(BaseModel):
    labels : list



@app.get("/")
def read_root():
    return {"Hello" : "World"}

def get_model():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('prakhars/instagram_caption_classifier')
    return tokenizer, model 

label_dict = {
11: 'Travel',
 0: 'Art and Culture',
 7: 'Music',
 3: 'Entertainment',
 8: 'Parenting',
 6: 'Health and Fitness',
 10: 'Sports',
 5: 'Food',
 2: 'Beauty, Fashion and Lifestyle',
 1: 'Automotive',
 9: 'Pets',
 4: 'Finance'
}




tokenizer, model = get_model()




@app.post('/predict/',response_model = EntityOut)
async def read_root(request: EntityIn, model=model, tokenizer=tokenizer, label_dict=label_dict):
    
    
    encodings = tokenizer(request.texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**encodings)
    predictions = outputs.logits.argmax(-1)
    out_codes = predictions.tolist()
    out_labels = {'labels':[]}
    for code in out_codes:
        out_labels['labels'].append(label_dict[code])

    

    return out_labels
    




