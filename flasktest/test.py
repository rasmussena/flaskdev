from flask import Flask, render_template_string, request

import os

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = openai_api_key

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Gets data from the pdf file.
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("data.pdf")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using data from the pdf file.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Custom template for RAG.
from langchain_core.prompts import PromptTemplate
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Be as concise as possible. You are a psychology professor and you are speaking to a student.
You are by-the-book and always prefer the material in the context over your own knowledge.

{context}

Question: {question}


Helpful Answer:
"""

prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

app = Flask(__name__)

def get_answer(question):
    info = ""
    for chunk in rag_chain.stream(question):
        info += chunk
    return info

@app.route('/')
def index():
    # Render the form for user input
    return render_template_string('''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>Question Form</title>
        </head>
        <body>
            <h1>Ask a Question:</h1>
            <form action="/ask" method="post">
                <label for="question">Enter your question:</label>
                <input type="text" id="question" name="question" required>
                <button type="submit">Submit</button>
            </form>
        </body>
        </html>
    ''')

@app.route('/ask', methods=['POST'])
def ask():
    # Get the question from the form
    question = request.form.get('question')
    # Generate the answer
    answer = get_answer(question)
    # Render the answer on a new page
    return render_template_string('''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>Answer</title>
        </head>
        <body>
            <h1>Question:</h1>
            <p>{{ question }}</p>
            <h2>Answer:</h2>
            <p>{{ answer }}</p>
            <a href="/">Ask another question</a>
        </body>
        </html>
    ''', question=question, answer=answer)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
