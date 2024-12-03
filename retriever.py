# from langchain.retrievers.document_compressors import DocumentCompressorPipeline
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_transformers import EmbeddingsRedundantFilter,LongContextReorder
# from langchain.retrievers.document_compressors import EmbeddingsFilter
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import CohereRerank
# from langchain_community.llms import Cohere
# from embedding import *

# def Vectorstore_backed_retriever(vectorstore,search_type="similarity",k=4,score_threshold=None):

#     search_kwargs={}
#     if k is not None:
#         search_kwargs['k'] = k
#     if score_threshold is not None:
#         search_kwargs['score_threshold'] = score_threshold

#     retriever = vectorstore.as_retriever(
#         search_type=search_type,
#         search_kwargs=search_kwargs
#     )
#     return retriever


# def create_compression_retriever(embeddings, base_retriever, chunk_size=500, k=16, similarity_threshold=None):

#     # 1. splitting documents into smaller chunks
#     splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, separator=". ")
    
#     # 2. removing redundant documents
#     redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

#     # 3. filtering based on relevance to the query    
#     relevant_filter = EmbeddingsFilter(embeddings=embeddings, k=k, similarity_threshold=similarity_threshold) # similarity_threshold and top K

#     # 4. Reorder the documents 
    
#     # Less relevant document will be at the middle of the list and more relevant elements at the beginning or end of the list.
#     # Reference: https://python.langchain.com/docs/modules/data_connection/retrievers/long_context_reorder
#     reordering = LongContextReorder()

#     # 5. Create compressor pipeline and retriever
    
#     pipeline_compressor = DocumentCompressorPipeline(
#         transformers=[splitter, redundant_filter, relevant_filter, reordering]  
#     )
#     compression_retriever = ContextualCompressionRetriever(
#         base_compressor=pipeline_compressor, 
#         base_retriever=base_retriever
#     )

#     return compression_retriever


# def CohereRerank_retriever(
#     base_retriever, 
#     cohere_api_key,cohere_model="rerank-multilingual-v2.0", top_n=8
# ):

#     compressor = CohereRerank(
#         cohere_api_key=cohere_api_key, 
#         model=cohere_model, 
#         top_n=top_n
#     )

#     retriever_Cohere = ContextualCompressionRetriever(
#         base_compressor=compressor,
#         base_retriever=base_retriever
#     )
#     return retriever_Cohere


# def retrieval_blocks(
#     vector_store,
#     embeddings,
#     retriever_type="Cohere_reranker",
#     base_retriever_search_type="similarity", base_retriever_k=10, base_retriever_score_threshold=None,
#     compression_retriever_k=16,
#     cohere_api_key="2XlmljR7odGXf3Gx21Vk2feKdC23hNFoCphij07P", cohere_model="rerank-multilingual-v2.0", cohere_top_n=8,
# ):

#     try:
#         base_retriever = Vectorstore_backed_retriever(
#             vector_store,
#             search_type=base_retriever_search_type,
#             k=base_retriever_k,
#             score_threshold=base_retriever_score_threshold
#         )

#         retriever = None
#         if retriever_type=="Vectorstore_backed_retriever": 
#             retriever = base_retriever
    
#         # 7. Contextual Compression Retriever
#         if retriever_type=="Contextual_compression":    
#             retriever = create_compression_retriever(
#                 embeddings=embeddings,
#                 base_retriever=base_retriever,
#                 k=compression_retriever_k,
#             )
    
#         # 8. CohereRerank retriever
#         if retriever_type=="Cohere_reranker":
#             retriever = CohereRerank_retriever(
#                 base_retriever=base_retriever, 
#                 cohere_api_key=cohere_api_key, 
#                 cohere_model=cohere_model, 
#                 top_n=cohere_top_n
#             )
    
#         return retriever
#     except Exception as e:
#         print(e)

from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import EmbeddingsRedundantFilter, LongContextReorder
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank # Commented out Cohere rerank
from langchain_community.llms import Cohere  # Commented out Cohere import
from embedding import *

def Vectorstore_backed_retriever(vectorstore, search_type="similarity", k=4, score_threshold=None):
    search_kwargs = {}
    if k is not None:
        search_kwargs['k'] = k
    if score_threshold is not None:
        search_kwargs['score_threshold'] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    return retriever


def create_compression_retriever(embeddings, base_retriever, chunk_size=500, k=16, similarity_threshold=None):
    # 1. splitting documents into smaller chunks
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, separator=". ")

    # 2. removing redundant documents
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

    # 3. filtering based on relevance to the query    
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, k=k, similarity_threshold=similarity_threshold)  # similarity_threshold and top K

    # 4. Reorder the documents 
    reordering = LongContextReorder()

    # 5. Create compressor pipeline and retriever
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter, reordering]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=base_retriever
    )

    return compression_retriever

# Commented out Cohere reranking-related function
def CohereRerank_retriever(
    base_retriever, 
    cohere_api_key, cohere_model="rerank-multilingual-v2.0", top_n=8
):
    compressor = CohereRerank(
        cohere_api_key=cohere_api_key, 
        model=cohere_model, 
        top_n=top_n
    )

    retriever_Cohere = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    return retriever_Cohere


def retrieval_blocks(
    vector_store,
    embeddings,
    cohere_api_key,
    retriever_type= "Vectorstore_backed_retriever"  ,   # "Vectorstore_backed_retriever",  # Default to vectorstore-backed retriever
    base_retriever_search_type="similarity", base_retriever_k=10, base_retriever_score_threshold=None,
    compression_retriever_k=16,
    #Cohere cohere_api_key="euNrotQ7k1wjnNpu2h5rv7Jc4sz9ckQFVNvMjVhw"
    cohere_model="rerank-multilingual-v2.0", cohere_top_n=8,
):
    print("inside retrival block")
    try:
        base_retriever = Vectorstore_backed_retriever(
            vector_store,
            search_type=base_retriever_search_type,
            k=base_retriever_k,
            score_threshold=base_retriever_score_threshold
        )

        retriever = None
        if retriever_type == "Vectorstore_backed_retriever":
            retriever = base_retriever

        # 7. Contextual Compression Retriever
        if retriever_type == "Contextual_compression":
            retriever = create_compression_retriever(
                embeddings=embeddings,
                base_retriever=base_retriever,
                k=compression_retriever_k,
            )

        if retriever_type == "Cohere_reranker":
            retriever = CohereRerank_retriever(
                base_retriever=base_retriever, 
                cohere_api_key=cohere_api_key, 
                cohere_model=cohere_model, 
                top_n=cohere_top_n
            )

        return retriever
    except Exception as e:
        print(e)
