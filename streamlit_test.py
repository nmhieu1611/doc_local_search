import streamlit as st
import tkinter as tk
from tkinter import filedialog as fd
from transformers import AutoTokenizer, AutoModel
import torch
import os.path
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.api.types import Documents, Embeddings
import PyPDF2
import nltk
from nltk.util import ngrams

model = SentenceTransformer('./gte-tiny')
chroma_client = chromadb.PersistentClient(path="./test_chromadb")

if 'document_list' not in st.session_state:
    st.session_state.document_list = [x.name for x in chroma_client.list_collections()]
if 'new_doc' not in st.session_state:
    st.session_state.new_doc = ""
if 'search_res' not in st.session_state:
    st.session_state.search_res = []

# def click_select_file_btn():
    # st.session_state.book_count += 1
    # st.session_state.new_doc = 'book ' + str(st.session_state.book_count)

def embed_function(texts: Documents) -> Embeddings:
    return model.encode(texts).tolist()
    
def click_ingest_file_btn():
    # creating a pdf file object
    pdfFileObj = open(st.session_state.new_doc, 'rb')
     
    # creating a pdf reader object
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
     
    # printing number of pages in pdf file
    print(len(pdfReader.pages))
    
    db_name = st.session_state.new_doc.split('/')[-1].split('.')[0].replace(' ','_')
    db_name = db_name[0:63] if len(db_name) > 63 else db_name
    
    
    db = chroma_client.get_or_create_collection(
        name=db_name,
        embedding_function=embed_function,
        metadata={"hnsw:space": "cosine"}
    )

    for i in range(len(pdfReader.pages)):
        # creating a page object
        pageObj = pdfReader.pages[i]
         
        # extracting text from page
        
        print(str('*' * 50), "page ", i, str('*' * 50))
        # print(pageObj.extract_text())
        
        tokenize = nltk.word_tokenize(pageObj.extract_text())
        chunksize = 100
        overlap = 35
        bigrams = ngrams(tokenize, chunksize)

        bigrams = list(bigrams)
        if len(bigrams) == 0:
            continue
        
        page_documents = [" ".join(x) for x in bigrams[::chunksize - overlap]]
        page_ids = ['p' + str(i) + '_c' + str(j) for j in range(len(page_documents))]
        print(page_documents)
        print(page_ids)
        db.upsert(
            documents=page_documents,
            ids=page_ids
        )
     
    # closing the pdf file object
    pdfFileObj.close()
    st.session_state.document_list.append(db_name)
    
def click_search_btn():
    print("*****", query)
    n_results = 5
    res = db.query(query_texts=query, n_results=n_results)
    n_results = len(res['ids'][0])
    st.session_state.search_res = ['ID: ' + res['ids'][0][i] + "\t Score: " + str(res['distances'][0][i]) + "\n\n" + res['documents'][0][i] for i in range(n_results)]
    
# Set up tkinter
root = tk.Tk()
root.withdraw()

# Make folder picker dialog appear on top of other windows
root.wm_attributes('-topmost', 1)



st.header('Search your document')

# ###### side bar ######

x = st.sidebar.empty()
x.info("Ingest new document: ")
# st.sidebar.write("Ingest new document: " + st.session_state.new_doc)

select_file_btn = st.sidebar.button("Select file")
if select_file_btn:
    st.session_state.new_doc = fd.askopenfilename(master=root)
    x.info("Ingest new document: " + st.session_state.new_doc)

ingest_file_btn = st.sidebar.button("Ingest new file", on_click=click_ingest_file_btn)

st.sidebar.write("---")

target_doc = st.sidebar.selectbox(
     'Select the document:',
     tuple(st.session_state.document_list))

st.sidebar.write('You are searching in ', target_doc)

db = chroma_client.get_collection(
    name=target_doc,
    embedding_function=embed_function
)

# ###### search screen ######

query = st.text_input(label='Your query:')

search_btn = st.button("Search", on_click=click_search_btn)

for res in st.session_state.search_res:
    st.write(res)
    st.write("---")