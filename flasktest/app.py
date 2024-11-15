from flask import Flask, render_template, redirect, url_for, request, flash, render_template_string
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

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

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

from werkzeug.security import generate_password_hash, check_password_hash


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
with app.app_context():
    db.create_all()
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


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        new_user = User(email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created!', category='success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Login failed. Check your email and password.', 'danger')
    
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return f'Hello, {current_user.email}!'

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/admin/users')
def list_users():
    users = User.query.all()
    user_data = [(user.id, user.email) for user in users]
    return render_template('list_users.html', users=user_data)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

if __name__ == '__main__':
    app.run(host ='0.0.0.0', debug=True)