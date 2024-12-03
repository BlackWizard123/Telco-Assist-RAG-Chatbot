import json
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Blueprint, session
import os, re
import shutil
from pathlib import Path
from langchain_community.vectorstores import Chroma
import chromadb
import time

# Import your custom modules for embeddings, retriever, etc.
from embedding import *
from retriever import *
from convRetrChain import *

chat_bp = Blueprint('chat_bp', __name__)

ALLOWED_EXTENSIONS = {'pdf', 'csv', 'docx', 'txt'}
global memory, chain, upload_folder, vector_store, user_folder, username
global hfapi1, hfapi2, chreapi
global llm, condense_question_llm
memory = None
chain = None

# Functions

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_answer(user_question):
    global memory, chain
    print("in get answer")
    if not memory or not chain:
        return "Memory or Chain not initialized. Please upload documents first."

    # chat_history = secondary_memory.chat_memory 
    chat_history = memory.chat_memory 
    result = chain.invoke({
        'question': user_question,
        'chat_history': chat_history
    })
    return result['answer']

def load_vectorDB_from_chroma_db(db_path):
    global hfapi2, chreapi
    embeddings = select_embeddings_model(hfapi2)
    chroma_client = Chroma( embedding_function= embeddings , persist_directory= str(db_path))
    retriever = retrieval_blocks(vector_store=chroma_client, embeddings=embeddings, cohere_api_key= chreapi)
    return retriever

def sanitizeFolderName(fileName):
    return re.sub(r'[^\w\s.-]', '', fileName).strip()

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '', filename)

def generate_topic(question, answer):
    # Create a topic-generation-specific prompt
    topic_prompt_text = f"""Based on the following conversation, generate a single, concise, creative topic with only 3 to 5 
    words for naming the chat. Use only alphabets, numbers, hyphens, underscores, and full stops. 
    Do not exceed 45 characters or end with a full stop. Follow the strict order of not to use more than 5 words!
    
Question: {question}
Answer: {answer}

Topic:"""

    # Directly call the LLM with the prompt
    topic_result = condense_question_llm.invoke(topic_prompt_text)
    topic_result = sanitize_filename(topic_result)
    topic_result = topic_result.split('\n', 1)[0][:40] + " " + time.strftime("(%d-%m  %H.%M.%S)")

    print (topic_result, end = "\n\n\n\n")

    return topic_result

#-------------------------------------------------------
# Routes

@chat_bp.route('/chat')
def chat_index():
    global llm, condense_question_llm
    global username, hfapi1, hfapi2, chreapi, user_folder
    username = session.get('username')
    hfapi1 = session.get('HFAPI1')
    hfapi2 = session.get('HFAPI2')
    chreapi = session.get('CHREAPI')
    user_folder = Path(__file__).resolve().parent.joinpath("userdata", username)
    llm = LLM(api_key = hfapi1 )
    condense_question_llm = LLM(api_key = hfapi2)
    return render_template('Chat _ Version 1.html')

@chat_bp.route('/upload', methods=['POST'])
def upload_and_create_embeddings():
    global chain, memory, user_folder  # Declare these as global to modify them
    global llm, condense_question_llm

    # Retrieve the vector DB name from the form data
    Vector_DB_Name = request.form.get("vecname", "")
    if not Vector_DB_Name:
        flash('Vector DB name is required!')
        return redirect(request.url)

    # Update UPLOAD_FOLDER and VECTOR_STORE based on the vector DB name
    user_folder = Path(__file__).resolve().parent.joinpath("userdata", username)
    upload_folder = Path(__file__).resolve().parent.joinpath("userdata", username, "Vector Store", Vector_DB_Name, "Documents")
    vector_store = Path(__file__).resolve().parent.joinpath("userdata",username, "Vector Store", Vector_DB_Name)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    chat_file = user_folder.joinpath("Chat History",f"{Vector_DB_Name}_{timestamp}_chat.json")
    session['current_vector_db'] = Vector_DB_Name
    session['current_chat_file'] = str(chat_file)

    # Create necessary directories
    upload_folder.mkdir(parents=True, exist_ok=True)
    vector_store.mkdir(parents=True, exist_ok=True)


    # Initialize the new JSON file with vector_db reference
    with open(chat_file, 'w') as f:
        json.dump({"vector_db": Vector_DB_Name, "created_on": time.strftime("%d-%m-%Y--%H:%M:%S"), "conversations": []}, f)

    # Validate and process uploaded files
    if 'files' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    files = request.files.getlist('files')
    valid_files = False

    uploaded_filenames = []
    for file in files:
        if file.filename == '':
            continue
        
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = upload_folder.joinpath(filename)
            file.save(file_path)
            valid_files = True
            uploaded_filenames.append(filename)
        else:
            flash(f'File type not allowed: {file.filename}')

    if valid_files:
        try:
            documents = langchain_document_loader(upload_folder)
            chunks = split_documents_to_chunks(documents)
            embeddings = select_embeddings_model(hfapi2)
            vectorDB = create_vectorDB(chunks, embeddings, Vector_DB_Name, vector_store)
            retriever = load_vectorDB_from_chroma_db(vector_store.joinpath("ChromaDB"))
            # retriever = retrieval_blocks(vector_store=vectorDB, embeddings=embeddings)
            if retriever:
                
                flash("Files uploaded and vector DB created successfully! Retriever initiated!")
                chain, memory = create_ConversationalRetrievalChain(
                    llm=llm,
                    condense_question_llm=condense_question_llm,
                    retriever=retriever
                )
                # chain, memory = custom_ConversationalRetrievalChain(
                #     llm=llm,
                #     condense_question_llm=condense_question_llm,
                #     retriever=retriever
                # )
                if chain and memory:
                    flash("Memory chain initiated... Now you can start chatting!")
            else:
                print("no")
                flash("Failed to initialize the retriever or memory chain")
        except Exception as e:
            flash(f'Error processing files: {str(e)}')
    else:
        flash('No valid files were uploaded')

    return jsonify({'files': uploaded_filenames, "db_name": Vector_DB_Name}), 200

@chat_bp.route('/load_vector_db_with_chat', methods=['POST'])
def loadVectorDBwithChat():
    global chain, memory, user_folder
    global llm, condense_question_llm
    chain = None
    memory = None
    dbName = request.form.get('dbName')
    chatFile = request.form.get('chatFile')
    dbName = sanitizeFolderName(dbName)
    print(chatFile, dbName)
    user_db_path = Path(__file__).resolve().parent.joinpath("userdata", username, "Vector Store", dbName,"ChromaDB")
    user_docs_path = Path(__file__).resolve().parent.joinpath("userdata", username, "Vector Store", dbName,"Documents")
    print(user_db_path)

    # Check if the folder exists
    if not os.path.exists(user_db_path):
        print("not exist")
        print(str(user_db_path) + "   " + str(dbName))
        return jsonify({'status': 'vectorDB_unavailable', 'db_name': dbName})
    else:
        files = os.listdir(os.path.join(user_docs_path))
        retriever = load_vectorDB_from_chroma_db(user_db_path)
        if retriever:
            chat_file_path = user_folder.joinpath("Chat History", chatFile)
            session['current_vector_db'] = dbName
            session['current_chat_file'] = str(chat_file_path)
            print("Vector DB loaded and retriever initiated successfully!")
            
            chain, memory = create_ConversationalRetrievalChain(
                llm=llm,
                condense_question_llm=condense_question_llm,
                retriever=retriever
            )
            if chain and memory:
                print("memory is there " + str(user_db_path) + "   " + str(dbName))
                return jsonify({'status': 'success',"db_name": dbName, "files": files})
            else:
                return jsonify({'status': 'memory_chain_failure'})
        else:
            return jsonify({'status': 'retriever_initialization_failure'})
    
@chat_bp.route('/load_vector_db', methods=['POST'])
def load_vector_db():
    global chain, memory, user
    global llm, condense_question_llm
    chain = None
    memory = None
    folder_name = request.form.get('folder')
    print(folder_name)
    folder_name = sanitizeFolderName(folder_name);
    print(folder_name)
    user_db_path = Path(__file__).resolve().parent.joinpath("userdata", username, "Vector Store", folder_name,"ChromaDB")
    user_docs_path = Path(__file__).resolve().parent.joinpath("userdata", username, "Vector Store", folder_name,"Documents")
    print(user_db_path)
    files = os.listdir(os.path.join(user_docs_path))



    # Check if the folder exists
    if not os.path.exists(user_db_path):
        print("not exist")
        print(str(user_db_path) + "   " + str(folder_name))
    
    retriever = load_vectorDB_from_chroma_db(user_db_path)
    if retriever:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        chat_file = user_folder.joinpath("Chat History",f"{folder_name}_{timestamp}_chat.json")
        session['current_vector_db'] = folder_name
        session['current_chat_file'] = str(chat_file)

        # Initialize the new JSON file with vector_db reference
        with open(chat_file, 'w') as f:
            json.dump({"vector_db": folder_name, "created_on": time.strftime("%d-%m-%Y--%H:%M:%S"), "conversations": []}, f)

        print("Vector DB loaded and retriever initiated successfully!")
        
        chain, memory = create_ConversationalRetrievalChain(
            llm=llm,
            condense_question_llm=condense_question_llm,
            retriever=retriever
        )
        if chain and memory:
            print("memory is there " + str(user_db_path) + "   " + str(folder_name))
            # if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # # Delete the folder and its contents
            # shutil.rmtree(folder_path)
            return jsonify({'status': 'success',"db_name": folder_name, "files": files})
        else:
            return jsonify({'status': 'memory_chain_failure'})
    else:
        return jsonify({'status': 'retriever_initialization_failure'})
    
    # try:
    #     retriever = load_vectorDB_from_chroma_db(user_db_path)

    #     if retriever:
    #         print("Vector DB loaded and retriever initiated successfully!")
    #         # Initialize chain and memory with the loaded retriever
    #         chain, memory = create_ConversationalRetrievalChain(
    #             llm=llm,
    #             condense_question_llm=condense_question_llm,
    #             retriever=retriever
    #         )
    #         if chain and memory:
    #             print("memory is there")
    #             return jsonify({'status': 'success'})
    #         else:
    #             return jsonify({'status': 'memory_chain_failure'})
    #     else:
    #         return jsonify({'status': 'retriever_initialization_failure'})

    # except Exception as e:
    #     return jsonify({'status': 'vectordb_missing'})

@chat_bp.route('/load_public_vector_db', methods=['POST'])
def load_public_vector_db():
    global chain, memory
    global llm, condense_question_llm
    chain = None
    memory = None
    folder_name = request.form.get('folder')
    folder_name = sanitizeFolderName(folder_name);
    user_db_path = Path(__file__).resolve().parent.joinpath("userdata", "adminOpenDB", folder_name, "ChromaDB")
    print(user_db_path)
    # Check if the folder exists
    if not os.path.exists(user_db_path):
        print("not exist")
        print(str(user_db_path) + "   " + str(folder_name))
    
    retriever = load_vectorDB_from_chroma_db(user_db_path)
    if retriever:
        print("Vector DB loaded and retriever initiated successfully!")
        
        chain, memory = create_ConversationalRetrievalChain(
            llm=llm,
            condense_question_llm=condense_question_llm,
            retriever=retriever
        )
        if chain and memory:
            # print("memory is there " + str(user_db_path) + "   " + str(folder_name))
            # if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # # Delete the folder and its contents
            # shutil.rmtree(folder_path)
            return jsonify({'status': 'success',"db_name": folder_name,})
        else:
            return jsonify({'status': 'memory_chain_failure'})
    else:
        return jsonify({'status': 'retriever_initialization_failure'})
    
@chat_bp.route('/delete_folder', methods=['POST'])
def delete_folder():
    folder_name = request.form.get('folder')
    user_folder = Path(__file__).resolve().parent.joinpath("userdata", username)
    
    # Create the full path to the folder
    folder_path = os.path.join(user_folder, "Vector Store", folder_name)
    
    try:
        # Check if the folder exists
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # Delete the folder and its contents
            shutil.rmtree(folder_path)
            return jsonify(success=True, message=f'Folder "{folder_name}" deleted successfully.')
        else:
            return jsonify(success=False, message=f'Folder "{folder_name}" does not exist.'), 404
    except Exception as e:
        # Handle exceptions (like permission issues, etc.)
        return jsonify(success=False, message=str(e)), 500
    
@chat_bp.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    file_path = os.path.join(upload_folder, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({'success': f'{filename} deleted successfully'}), 200
    else:
        return jsonify({'error': 'File not found'}), 404

@chat_bp.route('/ask_question', methods=['POST'])
def ask_question():
    global memory, user_folder
    
    if not memory:
        return jsonify({"response": "Memory is not initialized. Please upload documents first."})
    
    user_question = request.form.get('user_input').strip().lower()
    response = get_answer(user_question)
    last_answer_start = response.rfind("Answer:") + len("Answer:")
    last_answer = response[last_answer_start:].strip() 

    chat_file = session.get('current_chat_file')

    if  os.path.exists(chat_file):
        with open(chat_file, 'r') as f:
            chat_data = json.load(f)

        if len(chat_data['conversations']) == 0:
            # Generate a topic for the file name based on the first question and answer
            topic_context = generate_topic(user_question, last_answer)
            new_chat_file = user_folder.joinpath("Chat History",f"{topic_context}_chat.json") 
            
            # Rename the file
            os.rename(chat_file, new_chat_file)
            
            # Update the session with the new file name
            session['current_chat_file'] = str(new_chat_file)
            chat_file = new_chat_file
            

    chat_data['conversations'].append({
        "question": user_question,
        "answer": last_answer
    })

    with open(chat_file, 'w') as f:
        json.dump(chat_data, f)
    
    memory.save_context(
        inputs={'question': user_question},
        outputs={'answer':  last_answer }
    )
    return jsonify({"response": last_answer})

@chat_bp.route('/get_folders', methods=['GET'])
def get_folders():
    print("Accessing get_folders route")
    try:
        user_folder = Path(__file__).resolve().parent.joinpath("userdata", username, "Vector Store")
        folders = [folder for folder in os.listdir(user_folder) if os.path.isdir(os.path.join(user_folder, folder))]
        print(user_folder)
        return jsonify({'folders': folders}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@chat_bp.route('/get_vector_library', methods=['GET'])
def get_vector_library():
    print("Accessing get_folders route")
    try:
        user_folder = Path(__file__).resolve().parent.joinpath("userdata", "adminOpenDB")
        folders = [folder for folder in os.listdir(user_folder) if os.path.isdir(os.path.join(user_folder, folder))]
        print(user_folder)
        return jsonify({'folders': folders}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/load_chat_history', methods=['GET'])
def load_chat_history():
    print("reached chat history")
    chat_history_folder = user_folder.joinpath("Chat History")
    print(chat_history_folder, end= "++++++++\n\n")
    chat_files = [f for f in os.listdir(chat_history_folder) if f.endswith('_chat.json')]
    chat_histories = {}
    for file in chat_files: 
        with open(chat_history_folder.joinpath(file), 'r') as f:
            chat_histories[file] = json.load(f)
    return jsonify(chat_histories)

@chat_bp.route('/load_chat', methods=['GET'])
def load_chat():
    chat_file_name = request.args.get('file')
    chat_file = user_folder.joinpath("Chat History", chat_file_name)

    print(chat_file_name, end  = "\n\n")
    # Load the existing chat file and corresponding vector DB
    with open(chat_file, 'r') as f:
        chat_data = json.load(f)

    session['current_vector_db'] = chat_data['vector_db']  # Load corresponding vector DB
    session['current_chat_file'] = str(chat_file)  # Track the chat file in session

    return jsonify(chat_data)

@chat_bp.route('/logout', methods=['POST'])
def logout():
    # Clear the session
    session.clear()
    # Redirect to the login page
    return jsonify(success=True)# Return success response for AJAX