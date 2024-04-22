#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datasets import load_dataset #to load datasets from hf
from datasets import concatenate_datasets
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample, datasets, losses, SentenceTransformer, util, evaluation
from torch.utils.data import DataLoader


# In[2]:


# In[4]:


#upload med_wikidoc dataset
med_wikidoc = load_dataset("medalpaca/medical_meadow_wikidoc")

#remove answers
med_wikidoc = med_wikidoc.remove_columns(['instruction'])

# concatenate splits
med_wikidoc = concatenate_datasets([med_wikidoc["train"]])

#rename column name
med_wikidoc = med_wikidoc.rename_column('input','question')
med_wikidoc = med_wikidoc.rename_column('output','answer')



# In[6]:


meadow_dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")

#remove column
meadow_dataset=meadow_dataset.remove_columns(['instruction'])

# concatenate splits
meadow_dataset = concatenate_datasets([meadow_dataset["train"]])

#rename column name
meadow_dataset = meadow_dataset.rename_column('input','question')
meadow_dataset = meadow_dataset.rename_column('output','answer')


# In[8]:


MedQuad_dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset")

MedQuad_dataset =MedQuad_dataset.remove_columns(['qtype'])

MedQuad_dataset = concatenate_datasets([MedQuad_dataset["train"]])

#rename column name
MedQuad_dataset = MedQuad_dataset.rename_column('Question','question')
MedQuad_dataset = MedQuad_dataset.rename_column('Answer','answer')


# In[10]:


patient_dataset = load_dataset("medalpaca/medical_meadow_wikidoc_patient_information")

#remove answers
patient_dataset = patient_dataset.remove_columns(['instruction'])

# concatenate splits
patient_dataset = concatenate_datasets([patient_dataset["train"]])

#rename column name
patient_dataset = patient_dataset.rename_column('input','question')
patient_dataset = patient_dataset.rename_column('output','answer')




# In[14]:


# Concatenate all of the datasets
new_ds = concatenate_datasets([patient_dataset,MedQuad_dataset,med_wikidoc,meadow_dataset])

# Verify the concatenated dataset


# In[15]:


#train test split
new_ds = new_ds.train_test_split(test_size=0.1,shuffle=True)

print(f"len({new_ds})")


# In[16]:


model = SentenceTransformer('BAAI/bge-small-en-v1.5')
model.max_seq_length = 128


# In[17]:


train_data=[]
def create_train_inputexample(example):
    question = example['question']
    answer  = example['answer']
    # Create an InputExample object and append it to the data list
    input_example = InputExample(texts=[str(question),str(answer)])
    train_data.append(input_example)
    return None

new_ds['train'].map(create_train_inputexample)


# In[18]:


test_question=[]
test_answer=[]
def create_test_inputexample(example):
    test_question.append(example['question'])
    test_answer.append(example['answer'])
    return None

new_ds['test'].map(create_test_inputexample)


# In[19]:


dev_evaluator = evaluation.TranslationEvaluator(test_question,test_answer, show_progress_bar=True, write_csv=True,batch_size=64)


# In[20]:


#define train loss and train loader
train_loss = losses.CachedMultipleNegativesRankingLoss(model,mini_batch_size=32)
train_loader = datasets.NoDuplicatesDataLoader(train_data, batch_size=512)


# In[21]:


eval_initial = model.evaluate(dev_evaluator)
print(f"Initial evaluation {eval_initial}")


# In[22]:


model.fit(train_objectives=[(train_loader, train_loss)],evaluation_steps=100,epochs=80, warmup_steps=100,
evaluator=dev_evaluator,use_amp=True,optimizer_params = {'lr': 24e-5},
output_path="./results/bge-small-en-v1.5_medical")


# In[ ]:
#model.save('./models/bge-small-en-v1.5')



