from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from decimal import *
import numpy as np
import spacy
import time

# Load model needed
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# validation_dataset = load_dataset("stsb_multi_mt", name="en", split="dev")
evaluation_dataset = load_dataset("stsb_multi_mt", name="en", split="test")

# print(len(validation_dataset))
print("Number of data used for evaluation: " + str(len(evaluation_dataset)) + " data")

def preprocessingText(text):
    # TEXT PREPROCESSING
    # Note: Tokenization is in the pretrained BERT model
    # Turn the text into lowercase
    text = text.lower()
    # Remove unnecessary whitespaces (tab, \n, etc.)
    clean_text = " ".join(text.split())
    # Lemmatization using Spacy
    clean_text = nlp(clean_text)
    clean_text = " ".join([token.lemma_ for token in clean_text])
    # Replace the sentence in sentences with the cleaned sentence
    return clean_text

def calculatePearsonCoefficient(predicted, actual):
    rho = np.corrcoef(predicted, actual)
    coefficient = rho[0][1]
    print("The correlation matrix:")
    print(rho)
    print("The Pearson correlation coefficient is : " + str(coefficient))

def mainFunction():
    answer = evaluation_dataset['sentence1']
    query = evaluation_dataset['sentence2']
    predictedScore = []
    for i in range(len(answer)):
        sentences = [preprocessingText(answer[i]), preprocessingText(query[i])]
        predictedScore.append(calculateSimilarity(sentences))
    calculatePearsonCoefficient(predictedScore, np.array(evaluation_dataset['similarity_score']))
    
def calculateSimilarity(sentences):
    sentence_embeddings = model.encode(sentences)
    # Calculate the cosine similarity to get the similarity score of all sentences (except the first one)...
    # ...compared to the first sentence (key answer)
    score = cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:])[0]
    # Rounding the similarity score to an integer for the final score of each answer/student
    # return round(Decimal(score[0]*100).quantize(Decimal('0.1'), rounding = ROUND_HALF_UP))

    # FOR TESTING ONLY
    # Convert the score from 0-1 scale to 0-5 scale to match the data set
    score[0] = score[0] * (5)
    return score[0]
    
# =====================================================
time_start = time.perf_counter()
mainFunction()
time_elapsed = (time.perf_counter() - time_start)
print("Computational time starting from text preprocessing: " + str(time_elapsed) + " seconds")
# =====================================================