from tkinter import *
from tkinter import filedialog
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from decimal import *
import spacy

# Load model needed
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
sentences = []

root = Tk()
root.title('Automatic Essay Grading AI')
root.geometry("500x500")

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

def open_text():
    text_file = filedialog.askopenfilename(title="Open File", filetypes=(("Text Files", "*.txt"),))
    text_file = open(text_file, 'r')
    stuff = text_file.readlines()

    for sentence in stuff:
        # Preprocess text before appending
        sentences.append(preprocessingText(sentence))
    
    my_text.insert(END, stuff)
    text_file.close()

def newWindow():
    newWdw = Toplevel(root)
    root.withdraw()
    newWdw.title("Result")
    newWdw.geometry("500x500")
    Label(newWdw, text="Result:").pack()
    # Calculate similarity for the answers
    result = calculateSimilarity()
    # Output the scores into the GUI
    showResult = Text(newWdw)
    index = 0
    for res in result:
        index += 1
        showResult.insert(END, "Student-" + str(index) + " score: " + str(int(res)) + '\n')
    showResult.pack()
    
def calculateSimilarity():
    sentence_embeddings = model.encode(sentences)
    # print(sentence_embeddings.shape)
    # Calculate the cosine similarity to get the similarity score of all sentences (except the first one)...
    # ...compared to the first sentence (key answer)
    result = cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:])[0]
    # Rounding the similarity score to an integer for the final score of each answer/student
    for i in range(len(result)):
        result[i] = int(round(Decimal(result[i]*100).quantize(Decimal('0.1'), rounding = ROUND_HALF_UP)))
    return result

my_text = Text(root, width=40, height=10)
my_text.pack(pady=20)

open_button = Button(root, text="Open Text File", command=open_text)
open_button.pack(pady=20)

open_button = Button(root, text="Result", command=newWindow)
open_button.pack(pady=20)

root.mainloop()