import time
import os
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi

def get_video_id(url):
    """Extracts the YouTube video ID from a URL."""
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return match.group(1) if match else None

def multi_query_retriever(question):
        query_gen_prompt = PromptTemplate(
            template="""Generate 3 different search queries based on the user question to retrieve relevant information.
Return only the queries as a simple list, one per line, without numbering.

User question: {question}"""
        )
        
        chain = query_gen_prompt | llm | StrOutputParser()
        queries_text = chain.invoke(question)
        queries = [q.strip() for q in queries_text.split('\n') if q.strip()]
        
        all_docs = []
        seen = set()
        
        for query_variant in queries:
            docs = base_retriever.invoke(query_variant)
            for doc in docs:
                doc_id = doc.page_content[:100]
                if doc_id not in seen:
                    all_docs.append(doc)
                    seen.add(doc_id)
        
        return all_docs
    
def main():
    video_url = 'https://www.youtube.com/watch?v=wN13YeqEaqk'
    query = 'what does the speaker say about people sticking to one model after using it once and liking it?'
    video_id = get_video_id(video_url)

    if not video_id:
        print("Invalid YouTube URL")
        return

    start = time.time()

    faiss_index_path = f"faiss_index_{video_id}"
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    if os.path.exists(faiss_index_path):
        print("Loading existing FAISS index...")
        vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        print(f"Index loaded in {time.time() - start:.2f} seconds")
    else:
        print("Creating new FAISS index...")
        transcript_start = time.time()
        try:
            yt_api = YouTubeTranscriptApi()
            transcript = yt_api.fetch(video_id)
        except Exception as e:
            print(f"Could not retrieve transcript: {e}")
            return
        
        print(f"Transcript fetched in {time.time() - transcript_start:.2f} seconds")

        document = ' '.join(snippet.text for snippet in transcript)
        print(f"Document length: {len(document)} characters")

        chunk_start = time.time()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_text(document)
        print(f"Created {len(docs)} chunks in {time.time() - chunk_start:.2f} seconds")

        vector_store = FAISS.from_texts(docs, embedding=embeddings)
        vector_store.save_local(faiss_index_path)
        print(f"New index created and saved in {time.time() - chunk_start:.2f} seconds")

    base_retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    
    llm = OllamaLLM(model="llama3.2:3b")
    
    # uncomment to generate multiple query variations:
    #retriever = multi_query_retriever
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})


    def format_context(docs):
        return '\n'.join(doc.page_content for doc in docs)

    answer_prompt = PromptTemplate(
        template="""You are analyzing a YouTube video transcript. Answer questions based ONLY on the provided context.

Context from YouTube video transcript:
```{context}```

Question: {question}

Answer based only on the transcript content above. Be factual and specific."""
    )

    print("Generating answer with Ollama...")

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_context),
        'question': RunnablePassthrough()
    }) 

    main_chain = parallel_chain | answer_prompt | llm | StrOutputParser()

    response = main_chain.invoke(query)
    print("\nAnswer:")
    print(response)

    stop = time.time()
    print(f'\nTotal time: {stop - start:.2f} seconds')

if __name__ == '__main__':
    main()