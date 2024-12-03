from langchain.memory import ConversationSummaryBufferMemory,ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
# from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
# from langchain_openai import ChatOpenAI
from langchain.schema import format_document,Document
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser   # OutputParser
from operator import itemgetter
# from app_v1 import decrypt_api_key
from flask import session
 #mistralai/Mistral-7B-Instruct-v0.3
 

def LLM(api_key, LLM_provider="HuggingFace", temperature=0.5,top_p=0.95,model_name= "mistralai/Mistral-7B-Instruct-v0.2"):
    
    # if key_number == 1:
    #     # api_key = decrypt_api_key(session['HFAPI1'])
    #     api_key = session['HFAPI1']
    # elif key_number == 2:
    #     # api_key = decrypt_api_key(session['HFAPI2'])
    #     api_key = session['HFAPI2']

    if(api_key):
        print("api key retrieved from db")

    if LLM_provider == "OpenAI":
        llm = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            model_kwargs={
                "top_p": top_p
            }
        )
    if LLM_provider == "Google":
        llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model_name="gemini-pro",
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            convert_system_message_to_human=True
        )
    if LLM_provider == "HuggingFace":
        model_name="mistralai/Mistral-7B-Instruct-v0.2"
        llm = HuggingFaceEndpoint( 
            repo_id=model_name,                       # replace with your model name
            huggingfacehub_api_token=api_key,         # replace with your Hugging Face API token
            temperature=temperature,                  # control the randomness of responses
            top_p=top_p,                              # nucleus sampling parameter
            do_sample=True,                           # enable sampling for more varied outputs
            max_new_tokens=1024  
        )
    return llm

def create_memory(model_name='Google-gemini',memory_max_token=None):
    
    if model_name=="gpt-3.5-turbo":
        if memory_max_token is None:
            memory_max_token = 1024 # max_tokens for 'gpt-3.5-turbo' = 4096
        memory = ConversationSummaryBufferMemory(
            max_token_limit=memory_max_token,
            llm=ChatOpenAI(model_name="gpt-3.5-turbo",openai_api_key=openai_api_key,temperature=0.1),
            return_messages=True,
            memory_key='chat_history',
            output_key="answer",
            input_key="question"
        )
    else:
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key='chat_history',
            output_key="answer",
            input_key="question",
        )  
    return memory

def answer_template(language="english"):
    """Pass the standalone question along with the chat history and context 
    to the `LLM` wihch will answer"""
    
    template = f"""Answer the question at the end, using only the following context (delimited by <context></context>).

<context>

{{chat_history}}

{{context}} 
</context>

Question: {{question}}

"""
    return template

def create_ConversationalRetrievalChain(
    llm,condense_question_llm,
    retriever,
    chain_type= 'stuff',
    language="english",
    model_name = 'Google-gemini'
):
    """Create a ConversationalRetrievalChain.
    First, it passes the follow-up question along with the chat history to an LLM which rephrases 
    the question and generates a standalone query. 
    This query is then sent to the retriever, which fetches relevant documents (context) 
    and passes them along with the standalone question and chat history to an LLM to answer.
    """
    
    # 1. Define the standalone_question prompt. 
    # Pass the follow-up question along with the chat history to the `condense_question_llm`
    # which rephrases the question and generates a standalone question.

    standalone_question_prompt = PromptTemplate(
        input_variables=['chat_history', 'question'], 
        template="""Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in its original language.\n\n
Chat History:\n{chat_history}\n
Follow Up Input: {question}\n
Standalone question:""")

    # 2. Define the answer_prompt
    # Pass the standalone question + the chat history + the context (retrieved documents) to the `LLM` wihch will answer
    
    answer_prompt = ChatPromptTemplate.from_template(answer_template(language=language))

    # 3. Add ConversationSummaryBufferMemory for gpt-3.5, and ConversationBufferMemory for the other models
    
    memory = create_memory(model_name)
    print(".....working.....")

    # 4. Create the ConversationalRetrievalChain

    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=standalone_question_prompt,
        combine_docs_chain_kwargs={'prompt': answer_prompt},
        condense_question_llm=condense_question_llm,

        memory=memory,
        retriever = retriever,
        llm=llm,

        chain_type= chain_type,
        verbose= False,
        return_source_documents=True    
    )

    print("Conversational retriever chain created successfully!")
    
    return chain,memory

def _combine_documents(docs, document_prompt, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def custom_ConversationalRetrievalChain(
    llm,condense_question_llm,
    retriever,
    language="english",
    llm_provider="HuggingFace",
    model_name='Mistral-7b-Instruct',
):
    # 1. Create memory: ConversationSummaryBufferMemory for gpt-3.5, and ConversationBufferMemory for the other models
    
    memory = create_memory(model_name)

    # 2. load memory using RunnableLambda. Retrieves the chat_history attribute using itemgetter.
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"),
    )

    # 3. Pass the follow-up question along with the chat history to the LLM, and parse the answer (standalone_question).

    condense_question_prompt = PromptTemplate(
        input_variables=['chat_history', 'question'], 
        template = """Given the following conversation and a follow up question, 
            rephrase the follow up question to be a standalone question, in the same language as the follow up question.\n\n
            Chat History:\n{chat_history}\n
            Follow Up Input: {question}\n
            Standalone question:"""        
        )
    
    standalone_question_chain = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | condense_question_prompt
        | condense_question_llm
        | StrOutputParser(),
    }

    # 4. Combine load_memory and standalone_question_chain
    chain_question = loaded_memory | standalone_question_chain

    # 5. Retrieve relevant documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

        # 6. Get variables ['chat_history', 'context', 'question'] that will be passed to `answer_prompt`
    
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
    answer_prompt = ChatPromptTemplate.from_template(answer_template(language=language)) 
    # 3 variables are expected ['chat_history', 'context', 'question'] by the ChatPromptTemplate   
    answer_prompt_variables = {
        "context": lambda x: _combine_documents(docs=x["docs"],document_prompt=DEFAULT_DOCUMENT_PROMPT),
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history") # get it from `loaded_memory` variable
    }

    # 7. Load memory, format `answer_prompt` with variables (context, question and chat_history) and pass the `answer_prompt to LLM.
    # return answer, docs and standalone_question
    
    chain_answer = {
        "answer": loaded_memory | answer_prompt_variables | answer_prompt | llm,
        # return only page_content and metadata 
        "docs": lambda x: [Document(page_content=doc.page_content,metadata=doc.metadata) for doc in x["docs"]],
        "standalone_question": lambda x:x["question"] # return standalone_question
    }

    # 8. Final chain
    conversational_retriever_chain = chain_question | retrieved_documents | chain_answer

    print("Conversational retriever chain created successfully!")

    return conversational_retriever_chain,memory

