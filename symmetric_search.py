import re
import os
import gc
import json
import secrets
import random
import base64
import smtplib
import psycopg2
import numpy as np
import pandas as pd
from io import StringIO
from exception import *
from googleapiclient.errors import HttpError
from google.oauth2 import service_account
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer
from flask import Flask,request,render_template,jsonify,flash,redirect,url_for,session
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory,ConversationBufferWindowMemory
from chat_function import * 
from chromadb.utils import embedding_functions
import io
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.docstore import document 
import base64
import os 
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.document import Document
import time
from pypdf import PdfReader
import re
from uuid import uuid4
import copy
import chromadb
from langchain_community.document_loaders.text import TextLoader
load_dotenv()
google_api = os.getenv("GOOGLE_API_KEY")

google_embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=google_api)


def get_pdf_from_google_drive(file_id, credentials):
    """Fetch PDF file from Google Drive as BytesIO"""
    service = build('drive', 'v3', credentials=credentials)
    file_content = service.files().get_media(fileId=file_id).execute()
    return io.BytesIO(file_content)

gemini_model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key =google_api)
    
# used to store question answer
chat_memory = ConversationBufferWindowMemory(k=10,memory_key='chat_history',return_messages=True)
# chat_memory = ConversationSummaryBufferMemory(llm=gemini_model,memory_key='chat_history',return_messages=True)

SERVICE_ACCOUNT_FILE = './token_cred.json'
# Define scopes for Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive','https://www.googleapis.com/auth/spreadsheets']
# Shared credentials for all users
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)


app = Flask(__name__)
app.secret_key = secrets.token_hex(24)  # Required for flash messages

FOLDER_ID = '1SNEOT6spU3wD7AVJI_w53bfMDig2Ss4l' 

def get_files_from_google_drive():
    try:
        # Allowed file extensions
        allowed_extensions = ('.csv','.xlsx','.pdf')
        # credentials = authenticate_and_list_files()
        # Build Google Drive API client
        service = build('drive', 'v3', credentials=credentials)
        FOLDER_ID = '1SNEOT6spU3wD7AVJI_w53bfMDig2Ss4l' 
        try:
            folder_metadata = service.files().get(fileId=FOLDER_ID, fields='id').execute()
        except HttpError as e:
            if e.resp.status == 404:
                raise FolderNotAvailable()
            else:
                raise e  # Re-raise other HttpError exceptions

        # Query files within the folder
        query = f"'{FOLDER_ID}' in parents"
        if service:
            files = []
            next_page_token = None
            # Paginate through the results
            while True:
                response = service.files().list(
                    q=query,
               
                    spaces='drive',
                    fields='nextPageToken, files(id, name)',
                    pageToken=next_page_token
                ).execute()

                files.extend(response.get('files', []))
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
            # Print all files in the folder
            if files:
                files_with_id_dict = {}
                for file in files:
                    if file['name'].endswith(allowed_extensions):
                        files_with_id_dict[file['name']] = file['id']
                print('file:',files_with_id_dict)
                # # Get the list of files with allowed extensions
                # files = [os.path.basename(f) for f in [key for key,val in files_with_id_dict.items() if key.endswith(allowed_extensions)]]

                return files_with_id_dict
            else:
                raise FolderNotAvailable()
    except FolderNotAvailable as exe:
        # flash('FolderNotAvailable','error')
        return jsonify({'error': str(exe)}), 404


def sql_connection():
    "Establish a PostgreSQL connection."
    
    connection_string = 'postgres://postgres:postgres@localhost:5432/ghclMarketting'
    connection = psycopg2.connect(connection_string)
    return connection



def create_register_table(conn):
    try:
        sql_query = """ CREATE TABLE IF NOT EXISTS public.register_table
                        (
                            user_id serial PRIMARY KEY,
                            username varchar NOT NULL UNIQUE,
                            password bytea NOT NULL,
                            email_id character varying(255) NOT NULL UNIQUE
                        )
                    """
        curr = conn.cursor()
        curr.execute(sql_query)
        return 0
    except:
        return 'Table Not Created.'



def send_mail(receiver_email_id,message):
    try:
        sender_email_id = 'mayurnandanwar@ghcl.co.in'
        password = 'uvhr zbmk yeal ujhv'
        # creates SMTP session
        s = smtplib.SMTP('smtp.gmail.com', 587)
        # start TLS for security
        s.starttls()
        # Authentication
        s.login(sender_email_id, password)
        # message to be sent
        # sending the mail
        s.sendmail(sender_email_id, receiver_email_id, str(message))
        # terminating the session
        s.quit()

        del sender_email_id,password
        gc.collect()
        return 0
    except:
        return jsonify({'error':'The Message cannot be Sent.'})
    

def create_user_que_ans_table(curr,conn):
    sql_query = '''CREATE TABLE IF NOT EXISTS public.user_que_ans
                (
                    id serial PRIMARY KEY,
                    email_id varchar NOT NULL UNIQUE,
                    answer varchar,
                    feedback varchar
                )
                '''
    curr.execute(sql_query)
    return 0



@app.route('/')
def index():
    return render_template('login.html')



@app.route('/login', methods=['GET', 'POST'])
def login_page():
    try:
        if request.method == 'POST':
            email = request.form['email']
            password = request.form['password']
            
            print('email::',email)
            sql_query = f"select * from public.register_table where email_id = '{email}';"
            connection = sql_connection()
            if connection == 'Connection Error':
                raise PgConnectionError()
                
            else:
                curr = connection.cursor()
                curr.execute(sql_query)
                rows  = curr.fetchall()
                connection.close()

            if  len(rows) == 0 :
                flash("Email Id Not Found.", "error")

            if len(rows) != 0 :
                print('in')
                if rows[0][-1] != email :
                    flash("Invalid Email Id", "error")
                    return redirect(url_for('login_page'))
        
                decPassword = base64.b64decode(rows[0][-2]).decode("utf-8")
                if password == decPassword:

                    session['email'] = email
                    return redirect(url_for('chatpage'))
                else:
                    flash("Invalid Password", "error")
                del [decPassword]
                gc.collect()
            del email,password,sql_query,rows
            gc.collect()
        return render_template('login.html')
    except PgConnectionError as exe:
        return jsonify({'error':str(exe)}),400



@app.route('/signup', methods=['GET', 'POST'])
def signup():
    try:
        if request.method == 'POST':
            name = request.form['name']
            email = request.form['email']
            password = request.form['password']
            confirm_password = request.form['confirm_password']
            connection = sql_connection()
            table_create = create_register_table(connection)
            if table_create == 0:
                curr = connection.cursor()
                sql = f"SELECT email_id FROM public.register_table WHERE email_id = '{email}';"
                
                curr.execute(sql)
                rows = curr.fetchall()
                connection.close()
            else:
                raise PgConnectionError()
            
            if len(rows) == 0:
                # Check if passwords match
                if password != confirm_password:
                    flash("Passwords do not match!", "error")
                    return redirect(url_for('signup'))
                
                # Check password strength using regex
                password_pattern = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$')
                if not password_pattern.match(password):
                    flash("Password must contain at least 8 characters, including one uppercase letter, one lowercase letter, one digit, and one special character.", "error")
                    return redirect(url_for('signup'))
                
                # Generate a random token
                token = random.randint(100000, 999999)
                # Store the token in the session for validation
                session['token'] = str(token)
                session['email'] = email
                # this required for adding pass and name after validation
                session['password'] = password
                session['name'] = name
                # Send the token via email
                subject = "Email Verification Code"
                body = f"Your verification code is {token}. Please enter it on the website to verify your email."
                message = f"Subject: {subject}\n\n{body}"
                msg = send_mail(email, message)

                if msg == 0:
                    flash("Code has been sent to register email id.", "info")
                    # Redirect to the validate_mail route with email as a parameter
                    return redirect(url_for('validate_mail', email=email))
                del password_pattern,token,subject,body,message,msg
                gc.collect()
            else:
                flash("Email Already Exist.", "info")
        return render_template('signup.html')
    except PgConnectionError as exe:
        return jsonify({"error": str(exe)})


@app.route('/chatpage', methods=['GET'])
def chatpage():
    """Render the homepage with a list of allowed files."""

    if 'email' not in session:  # Check if user is logged in
            flash('Please login to access the chat page.', 'warning')  # Flash the message
            return redirect(url_for('login_page'))  # Redirect to login page
    
    file_dicts = get_files_from_google_drive()
    # files = list(file_dicts.keys())
    return render_template('chatpage.html', files=file_dicts)



def load_file_data(file_id, credentials,file_name):
    """Load data from the selected file based on its type (CSV, Excel, PDF)."""
    try:
        # Use the credentials passed into the function to authenticate the service
        service = build('drive', 'v3', credentials=credentials)
        file_metadata = service.files().get(fileId=file_id, fields="name, mimeType").execute()
        mime_type = file_metadata.get('mimeType')
        # Get the content of the file
        file_content = service.files().get_media(fileId=file_id).execute()
        
        # Check the file type
        if mime_type == 'application/vnd.ms-excel' or mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            # Get the content of the file
            loader = UnstructuredExcelLoader(file_path=None, file=io.BytesIO(file_content))
            data = loader.load() 

        elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            doc = Document(io.BytesIO(file_content))
            # Extract text from the document
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            # convert text to document
            data = [document.Document(page_content=text)]

        elif mime_type == 'application/pdf':
            print('mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm: pdf')
            data = extract_text_and_image_summary(file_content,file_name,gemini_model)
            # # convert text to document
            # data = [document.Document(page_content=text)]

        elif mime_type=='text/plain' or mime_type=='text/csv':
            # convert text to document
            data = [document.Document(file_content.decode('utf-8'))]


        del service,file_metadata,mime_type,file_content
        gc.collect()
        return data
     
    except Exception as e:
        return "Not Found"


@app.route('/ask', methods=['POST','GET'])
def get_ans_from_csv():
    ''' this function is used to get answer from given csv.

    Args:
    doc_file([CSV]): comma separated file 
    query_text :  Question

    Returns: Answer
    '''
    try:
        
        if 'email' not in session:
            app.logger.debug(f"Session data: {session.items()}")
            flash("Please log in to access this functionality.", "error")
            return redirect(url_for('login_page'))
            #return jsonify({"error": "Unauthorized. Please log in."}), 401

        email = session['email']
        print('email::',email)
        query_text = request.form.get("query_text")
        file_name = request.form.get('selected_file')
        print('file_name::',file_name)
        print('query_text:::',query_text)


        file_with_id_dict = get_files_from_google_drive()
        doc_id = file_with_id_dict[file_name]
        print('doc_id::',doc_id)

        if query_text:
            if not file_name or file_name == "Select a document":
                flash("Please select a document to proceed.")
                return redirect(url_for('chatpage'))

            persistant_dict = os.path.basename(file_name).split('.')[0].replace(' ', '_')
            if '&' in persistant_dict:
                persistant_dict = persistant_dict.replace('&','_')

            if not os.path.exists(persistant_dict):
                pdf_stream = get_pdf_from_google_drive(doc_id, credentials)
                extracted_text_file_name = os.path.splitext(file_name)[0] + '.txt'
                image_folder = os.path.splitext(file_name)[0] + '_images'
                print(extracted_text_file_name)
                if not os.path.exists(extracted_text_file_name):
                    extract_text_and_image_summary(
                        file_stream=pdf_stream,
                        extracted_text_file_name=extracted_text_file_name,
                        image_folder=image_folder,
                        gemini_model=gemini_model
                    )
            
                with open(extracted_text_file_name, mode='r') as f:
                    sap_data = f.read()

                chunk_text = split_text(sap_data)
                uuids = [str(uuid4()) for _ in range(len(chunk_text))]
                
                vector_store = Chroma.from_texts(texts=chunk_text,embedding=google_embeddings_model,persist_directory=persistant_dict,ids=uuids)
            else:
                print('else')
                vector_store = Chroma(persist_directory=persistant_dict, embedding_function=google_embeddings_model)

            retriever = vector_store.as_retriever()
            prompt = get_prompt()

            # Chain creation
            chain = get_chain(gemini_model, prompt, retriever, chat_memory)
            print(chain)

            response = chain.invoke({'question': query_text, 'chat_history': chat_memory})
            # print('response::', response['answer'])

            # # ✅ Format numbered answers into separate lines (e.g., "1. ... 2. ...")
            # answer = response['answer']
            # if isinstance(answer, str) and re.search(r'\d+\.', answer):
            #     parts = re.split(r'(?=\d+\.)', answer, maxsplit=1)
            #     if len(parts) == 2:
            #         header = parts[0].strip()
            #         numbered_section = parts[1]
            #         numbered_items = re.split(r'(?=\d+\.)', numbered_section)
            #         numbered_items = [item.strip() for item in numbered_items if item.strip()]
            #         formatted_numbered_text = '\n'.join(numbered_items)
            #         answer = f"{header}\n{formatted_numbered_text}"

            return jsonify({'answer': response['answer']})

    except Exception as e:
        print("Error:", e)
        return redirect(url_for('chatpage'))

    
@app.route('/validate_mail',methods=['POST','GET'])
def validate_mail():
    
    try:
        email = request.args.get('email')  # Retrieve email from query string
        
        if request.method == 'POST':
            entered_token = str(request.form['token'])

            # Compare the entered token with the session token
            if str(session.get('token')) == str(entered_token):
                password = session['password']
                name = session['name']
                encPassword = base64.b64encode(password.encode("utf-8"))
                connection = sql_connection()
                table_created = create_register_table(connection)
                if table_created==0:
                    sql_query = "INSERT INTO public.register_table (username, password, email_id) VALUES (%s, %s, %s);"
                    curr = connection.cursor()
                    curr.execute(sql_query, (name, encPassword, email))
                    connection.commit()
                    connection.close()
                else:
                    raise ConnectionError()

                #remove session after adding it to table 
                session.pop('password')
                session.pop('name')
                session.pop('token')

                flash("Signup successful! Please login.", "success")
                del password,name,encPassword,sql_query
                gc.collect()

                return redirect(url_for('login_page'))
            else:
                # return "Invalid token. Please try again.", 400
                flash("Invalid code. Please try again.", "error")  # Flash error message

        return render_template('validate_mail.html', email=email)
    
    except ConnectionError as exe:
        return jsonify({'error': str(exe)}),400
    
@app.route('/validate_mail_reset_password',methods=['POST','GET'])
def validate_mail_reset_password():
    email = request.args.get('email')  # Retrieve email from query string
    if request.method == 'POST':
        entered_token = str(request.form['token'])

        # Compare the entered token with the session token
        if str(session.get('reset_token')) == str(entered_token):
            return redirect(url_for('reset_password'))
        else:
            # return "Invalid token. Please try again.", 400
            flash("Invalid code. Please try again.", "error")  # Flash error message

    return render_template('reset_token_validate.html', email=email)


@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password(): 
    try:
        if request.method == 'POST':
            email = session['email']
            new_password = request.form['new_password']
            confirm_password = request.form['confirm_password']
            password_pattern = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$')
            if not password_pattern.match(new_password):
                flash("Password must contain at least 8 characters, including one uppercase letter, one lowercase letter, one digit, and one special character.", "error")
                return render_template('reset_password.html')
        
            if new_password != confirm_password:
                flash("Passwords do not match.", "error")
                return render_template('reset_password.html')
            # Check password strength using regex
            else:
                encPassword = base64.b64encode(new_password.encode("utf-8"))
                sql_query = "UPDATE public.register_table SET password = %s WHERE email_id = %s;"
                connection = sql_connection()
                if connection == 'Connection Error':
                    raise PgConnectionError()
                else:
                    curr = connection.cursor()
                    curr.execute(sql_query,(encPassword,email))
                    connection.commit()
                    connection.close()

                    flash("Password has been reset successfully. You can now log in.", "success")
                del encPassword,sql_query
                gc.collect()
                return redirect(url_for('login_page'))
        return render_template('reset_password.html')
    except PgConnectionError as exe:
        return jsonify({'error':str(exe)})
    

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    try:
        if request.method == 'POST':
            email = request.form['email']
            session['email'] = email
            sql_query = f"SELECT * FROM public.register_table WHERE email_id = '{email}';"
            connection = sql_connection()
            if connection == 'Connection Error':
                raise PgConnectionError()
            else:
                curr = connection.cursor()
                curr.execute(sql_query)
                rows = curr.fetchall()
                connection.close()

            if len(rows) == 0:
                flash("Email not found. Please SignUp", "error")
                return redirect(url_for('signup'))
            else:
                # Generate a random token
                reset_token = str(random.randint(100000, 999999))
                session['reset_token'] = reset_token
                subject = "Code For Password Change"
                body = f"Your verification code is {reset_token}. Please enter it on the website to verify your email."
                message = f"Subject: {subject}\n\n{body}"
                msg = send_mail(email, message)
                
                del reset_token,subject,body,message
                gc.collect()

                if msg == 0:
                    flash("Code has been sent to registered email id.", "info")
                    return redirect(url_for('validate_mail_reset_password', email=email))
                
            del email,sql_query,rows
            gc.collect()
        return render_template('forgot_password.html')
    
    except PgConnectionError as exe:
        return jsonify({'error':str(exe)})


@app.route('/clear', methods=['POST'])
def clear():
    """Clear user feedback or session data."""
    # Any session or data clearing logic goes here (if needed)

    # Redirect to index function
    return redirect(url_for('chatpage'))
    # return render_template('chatpage.html')


@app.route('/save_feedback', methods=['POST'])
def save_feedback():
    """Save user feedback."""
    email = session.get('email')
    feedback_data = request.json  # Expecting JSON data

    if not email:
        return jsonify({'error': 'User not logged in'}), 401

    # Ensure file name is retrieved correctly
    file_name = feedback_data.get('selected_file')  # Extract correctly

    feedback_res = {}
    feedback_res['question'] = feedback_data['question']
    feedback_res['feedback'] = feedback_data['feedback']

    if not file_name:
        return jsonify({'error': 'File name is missing'}), 400


    # Fetch existing feedback
    sql_query = """SELECT feedback FROM public.user_feedback WHERE email_id = %s AND file_name = %s;"""
    
    with sql_connection() as conn:
        with conn.cursor() as curr:
            curr.execute(sql_query, (email, file_name))
            res = curr.fetchone()  # Fetch a single row

    if not res:
        # Insert new feedback
        feedback_json = json.dumps([feedback_res])  # Store as list
        sql_query = """INSERT INTO public.user_feedback (email_id, file_name, feedback) 
                       VALUES (%s, %s, %s);"""
        with sql_connection() as conn:
            with conn.cursor() as curr:
                curr.execute(sql_query, (email, file_name, feedback_json))
                conn.commit()
    else:
        # ans_lst = json.loads(res[0])  # Convert JSON string to Python list
        ans_lst = res[0]
        ans_lst.append(feedback_res)  # Append new feedback
        feedback_json = json.dumps(ans_lst)  # Convert back to JSON

        sql_query = """UPDATE public.user_feedback SET feedback = %s 
                       WHERE email_id = %s AND file_name = %s;"""
        with sql_connection() as conn:
            with conn.cursor() as curr:
                curr.execute(sql_query, (feedback_json, email, file_name))
                conn.commit()

    return jsonify({'message': 'Feedback saved successfully'}), 200

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5015)




