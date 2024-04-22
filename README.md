# medical-qa

# medical-qa

## 🚀 Overview

This project focuses on developing a semantic search system for medical question answering utilizing pre-trained language models. The project utilizes datasets sourced from Hugging Face, containing medical question-answer pairs, which are processed to ensure compatibility with the chosen model architecture. The primary model employed for this task is the BAAI/bge-small-en-v1.5 model from the SentenceTransformers library.

## Motivation

This project addresses the critical need for efficient and accurate medical information retrieval systems. Traditional methods often struggle with the nuanced semantics of medical queries. By leveraging advanced natural language processing techniques and large-scale medical datasets, we aim to streamline the process, empowering healthcare professionals and patients with timely, reliable, and personalized medical information. Ultimately, our goal is to contribute to the efforts to make medical information more accessible.

## Dataset

This project incorporates multiple datasets sourced from the Hugging Face platform, each contributing to the training and evaluation of the semantic search system:

1. [keivalya/MedQuad-MedicalQnADataset](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset)
   - Description: This medical question-answer pairs intended for training machine learning models in the medical domain. It consists of three columns: 'qtype' denoting the 
   question type, 'Question' representing the medical queries, and 'Answer' providing corresponding answers. The dataset comprises 16,407 samples in the training split.
2. [medalpaca/medical_meadow_wikidoc](https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc)
   - Description: This dataset, consisting of 10,000 samples in the training split, contains medical information in the form of instructions, input, and output. The 'instruction' column provides contextual 
   information, while the 'input' and 'output' columns contain medical queries and their corresponding answers, respectively.
3. [medalpaca/medical_meadow_medical_flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)
   - Description: This dataset comprises 33,955 samples in the training split. It follows a similar structure to the medical_meadow_wikidoc dataset, with 'instruction', 'input', and 'output' columns. The '  
   instruction' column provides additional context, while the 'input' and 'output' columns contain medical questions and their corresponding answers.
4. [medalpaca/medical_meadow_wikidoc_patient_information](https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc_patient_information)
   - Description: With 5,942 samples in the training split, this dataset focuses on patient information within the medical domain. It features 'instruction', 'input', and 'output' columns similar to the previous 
   datasets, where 'input' represents medical queries and 'output' denotes corresponding answers.

After preprocessing, which involves removing irrelevant columns and renaming columns for uniformity, the datasets are concatenated into a single dataset. The resulting dataset contains 53,043 samples in the training split and 13,261 samples in the test split, with 'question' and 'answer' columns representing medical queries and their corresponding answers, respectively.

## Requirements:

 ```
datasets==2.18.0
numpy==1.24.4
pandas==2.0.3
sentence_transformers==2.5.1
torch==2.0.1

```
If you don't have the requirements installed, they can be installed with the following:

 ```
pip install -r requirements.txt

```



