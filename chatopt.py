import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from quart import Quart, request, jsonify, send_from_directory
import git
import pandas as pd
import re
import sqlite3
from datetime import datetime
from docx import Document
import torch

# Ensure the token is set correctly
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Clone or open the repository
def clone_repo(repo_url, repo_dir):
    if not os.path.exists(repo_dir):
        git.Repo.clone_from(repo_url, repo_dir)
    return git.Repo(repo_dir)

# Extract commit history
def get_commit_history(repo):
    commits = []
    for commit in repo.iter_commits():
        for file in commit.stats.files.keys():
            commits.append({
                'commit': commit.hexsha,
                'author': commit.author.name,
                'date': datetime.fromtimestamp(commit.committed_date),
                'message': commit.message,
                'file': file
            })
    return pd.DataFrame(commits)

# Analyze file content and names
def analyze_files(repo_dir):
    file_analysis = []

    keywords = {
        'feature': ['feature', 'add', 'implement'],
        'bugfix': ['fix', 'bug', 'resolve'],
        'refactor': ['refactor', 'clean', 'improve'],
        'documentation': ['doc', 'readme'],
        'test': ['test', 'unit test']
    }

    def categorize_content(content, filename):
        for category, kws in keywords.items():
            if any(re.search(rf'\b{k}\b', content, re.IGNORECASE) for k in kws) or \
               any(re.search(rf'\b{k}\b', filename, re.IGNORECASE) for k in kws):
                return category
        return 'other'

    for root, _, files in os.walk(repo_dir):
        for file in files:
            file_path = os.path.relpath(os.path.join(root, file), repo_dir)
            try:
                with open(os.path.join(root, file), 'r', errors='ignore') as f:
                    content = f.read()
                category = categorize_content(content, file)
                file_analysis.append({
                    'file': file_path,
                    'category': category
                })
            except Exception as e:
                print(f"Could not read file {file_path}: {e}")
                file_analysis.append({
                    'file': file_path,
                    'category': 'other'
                })

    return pd.DataFrame(file_analysis)

# Read Word documents and extract content
def read_word_docs(doc_dir):
    docs_data = []

    for root, _, files in os.walk(doc_dir):
        for file in files:
            if file.endswith(".docx"):
                file_path = os.path.relpath(os.path.join(root, file), doc_dir)
                try:
                    doc = Document(os.path.join(root, file))
                    content = "\n".join([para.text for para in doc.paragraphs])
                    docs_data.append({
                        'file': file_path,
                        'content': content
                    })
                except Exception as e:
                    print(f"Could not read file {file_path}: {e}")
                    docs_data.append({
                        'file': file_path,
                        'content': ''
                    })

    return pd.DataFrame(docs_data)

# Connect to the database
def connect_db(db_path):
    conn = sqlite3.connect(db_path)
    return conn

# Fetch emails from the database
def fetch_emails(conn):
    query = "SELECT sender, receiver, subject, body, timestamp FROM emails"
    emails = pd.read_sql_query(query, conn)
    return emails

# Process and categorize emails to identify work-related ones
def categorize_emails(emails):
    work_related_keywords = [
        'project', 'deadline', 'meeting', 'report', 'update', 'task',
        'review', 'migration', 'ui/ux', 'security', 'api', 
        'performance', 'deployment', 'testing', 'bug', 'feature'
    ]
    emails['is_work_related'] = emails['body'].apply(lambda x: any(keyword in x.lower() for keyword in work_related_keywords))
    return emails[emails['is_work_related']]

# Extract relevant details
def extract_details(emails):
    emails['operation'] = emails['subject'].apply(lambda x: re.findall(r'(project|deadline|meeting|report|update|task|review|migration|ui/ux|security|api|performance|deployment|testing|bug|feature)', x.lower()))
    emails['operation'] = emails['operation'].apply(lambda x: x[0] if x else 'other')
    emails['timestamp'] = pd.to_datetime(emails['timestamp'])
    emails.rename(columns={'timestamp': 'date', 'body': 'message'}, inplace=True)
    emails['commit'] = ''
    emails['author'] = emails['sender']
    emails['file'] = ''
    return emails

# Combine git, email, and doc data
def combine_data(git_df, email_df, doc_df):
    git_df['source'] = 'git'
    email_df['source'] = 'email'
    doc_df['source'] = 'doc'
    doc_df['author'] = ''
    doc_df['receiver'] = ''
    doc_df['commit'] = ''
    doc_df['message'] = doc_df['content']
    doc_df['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    combined_df = pd.concat([git_df, email_df, doc_df[['file', 'message', 'source', 'author', 'receiver', 'commit', 'date']]], ignore_index=True)
    return combined_df

# Infer objectives from commit messages and file analysis
def infer_objectives(df, file_analysis_df):
    df = df.merge(file_analysis_df, on='file', how='left')
    df['objective'] = df.apply(lambda row: row['category'] if pd.notnull(row['category']) else 'other', axis=1)
    return df

# Load and preprocess data
def load_data():
    repo_url = 'https://github.com/korrenhannes/website.git'
    repo_dir = '/Users/korrenhannes/Documents/GitHub/website'  # Replace with the local path to clone the repo
    repo = clone_repo(repo_url, repo_dir)
    commits_df = get_commit_history(repo)
    file_analysis_df = analyze_files(repo_dir)
    commits_df = infer_objectives(commits_df, file_analysis_df)

    db_path = '/Users/korrenhannes/Documents/GitHub/logger/dummy_emails.db'
    conn = connect_db(db_path)
    emails = fetch_emails(conn)
    work_emails = categorize_emails(emails)
    detailed_emails = extract_details(work_emails)
    conn.close()

    doc_dir = '/Users/korrenhannes/Documents/GitHub/logger/WordDocs'  # Replace with the directory containing Word documents
    docs_df = read_word_docs(doc_dir)

    combined_df = combine_data(commits_df, detailed_emails, docs_df)
    return combined_df

# Preload data and model
data = load_data()

# Create a full-text search index using SQLite FTS5
def create_search_index(data, db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS documents USING fts5(file, message, source, author, receiver, \"commit\", date)")
    
    # Convert timestamp to string
    data['date'] = data['date'].astype(str)

    # Insert data into the virtual table
    for _, row in data.iterrows():
        c.execute("INSERT INTO documents (file, message, source, author, receiver, \"commit\", date) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                  (row['file'], row['message'], row['source'], row['author'], row['receiver'], row['commit'], row['date']))
    
    conn.commit()
    return conn

db_path = "/Users/korrenhannes/Documents/GitHub/logger/search_index.db"
create_search_index(data, db_path)

# Load pre-trained model and tokenizer
model_name = "facebook/opt-1.3b"
token = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# Set device to cuda if available, otherwise use cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
    model.gradient_checkpointing_enable()
    model.to(device)
except RuntimeError as e:
    print(f"Failed to load model on CUDA due to: {e}. Falling back to CPU.")
    device = "cpu"
    model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

# Create a function to generate response
async def generate_response(user_input):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT * FROM documents WHERE message MATCH ?", (user_input,))
    relevant_data = pd.DataFrame(c.fetchall(), columns=['file', 'message', 'source', 'author', 'receiver', 'commit', 'date'])

    chatty_response = pipe(user_input, max_length=256, do_sample=True, temperature=0.6, top_p=0.9, truncation=True)[0]['generated_text']

    if not relevant_data.empty:
        response_data = []
        for _, row in relevant_data.iterrows():
            item = {
                'title': row['message'],
                'source': row['source'],
                'details': [
                    {'label': 'From', 'value': row['author']},
                    {'label': 'Date', 'value': row['date']},
                    {'label': 'To', 'value': row['receiver'] if row['source'] == 'email' else ''},
                    {'label': 'File', 'value': row['file']},
                    {'label': 'Commit', 'value': row['commit']}
                ]
            }
            response_data.append(item)

        # Generate summary
        summary = f"Received {len(response_data)} messages. "
        for item in response_data:
            summary += f"From: {item['details'][0]['value']}, Date: {item['details'][1]['value']}, Source: {item['source']}. "

        summary = summary[:100] + '...'  # Limit to 100 words

        detailed_response = {
            'type': 'timeline',
            'response': response_data,
            'chatty_response': chatty_response + "\n\nSummary: " + summary
        }
        return detailed_response
    else:
        return {'type': 'text', 'response': chatty_response}

# Deploy the chatbot
app = Quart(__name__)

@app.route("/chat", methods=["POST"])
async def chat():
    user_input = (await request.json).get("message")
    response = await generate_response(user_input)
    return jsonify(response)

@app.route('/')
async def index():
    return await send_from_directory('.', 'index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
