# YouTube RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for YouTube videos. It fetches a video transcript, splits it into chunks, stores embeddings in a FAISS index, retrieves the most relevant transcript chunks for a question, and generates an answer using an Ollama local LLM.

## Code overview

### 1) LLM initialization

```python
llm = OllamaLLM(model="llama3.2:3b")
```

Explanation:
This line creates the language model instance used for both query generation and final answer generation. It uses the local Ollama model llama3.2:3b.

### 2) Multi-query retriever function signature

```python
def multi_query_retriever(question):
```

Explanation:
This function takes a single user question, creates multiple alternate search queries from it, retrieves context with each query, and merges the retrieved documents.

### 3) Prompt template for query expansion

```python
query_gen_prompt = PromptTemplate(
	template="""Generate 3 different search queries based on the user question to retrieve relevant information.
Return only the queries as a simple list, one per line, without numbering.

User question: {question}"""
)
```

Explanation:
This prompt asks the model to generate three paraphrased search queries from the original question. Multiple query variants improve retrieval coverage when one phrasing misses relevant transcript chunks.

### 4) Query generation chain

```python
chain = query_gen_prompt | llm | StrOutputParser()
```

Explanation:
This composes a small LangChain pipeline where the prompt is sent to the LLM, and the raw output is converted to plain text by StrOutputParser. The resulting text is then split into separate query lines for retrieval.


## Usage
`python chatbot.py`

