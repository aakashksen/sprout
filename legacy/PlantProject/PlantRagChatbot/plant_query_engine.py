import requests
import json
from dotenv import load_dotenv
import os
import sys

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Add the project root to the Python path
sys.path.insert(0, PROJECT_ROOT)

# Load environment variables from root .env file
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# Import necessary libraries for embeddings and vector database
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class PlantQueryEngine:
    CHROMA_PATH = "chroma"
    
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
   
    {context}
   
    ---
    Important rules for all responses:
    1. If the user just says a greeting like "hi", "hello", or "yo", ask them what they'd like to know about plant care or diseases in the context of the plant at hand. 
    2. For actual plant questions, be concise and focus only on what was specifically asked.
    3. Do not provide lengthy step-by-step plans unless explicitly requested.
    4. For plant diseases, focus on the specific ailment and practical solutions for pot plants.
    5. Answer in a paragraph form without percentage numbers.
   
    User input: {question}
    """

    DIRECT_PROMPT_TEMPLATE = """
    You are an expert in plant care and diseases. Follow these important rules:
    1. If the user just says a greeting like "hi", "hello", or "yo", ask them what they'd like to know about plant care or diseases in the context of the plant at hand.
    2. For actual plant questions, be concise and focus only on what was specifically asked.
    3. Do not provide lengthy step-by-step plans unless explicitly requested.
    4. Focus on practical solutions for home gardeners growing plants in pots or small gardens.
    5. If the question is about a disease, explain identification and specific treatments concisely.

    User input: {question}
    """

    def __init__(self, model_name="deepseek-r1:14b", similarity_threshold=0.7, max_results=12):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        self.embeddings = None
        self.db = None
        
        # Try to initialize HuggingFace embeddings and Chroma database
        try:
            print("Initializing HuggingFace embeddings...")
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            
            # Load the Chroma database
            print(f"Loading Chroma database from {self.CHROMA_PATH}...")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            chroma_path = os.path.join(script_dir, self.CHROMA_PATH)
            self.db = Chroma(persist_directory=chroma_path, embedding_function=self.embeddings)
        except Exception as e:
            print(f"Warning: Could not initialize embeddings or database: {str(e)}")
            print("The chatbot will function in direct query mode only.")
            self.db = None

    def get_deepseek_completion(self, prompt, stream=False):
        """
        Query the DeepSeek API for a completion.
        """
        import os
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise EnvironmentError("DEEPSEEK_API_KEY environment variable not set.")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": stream
        }
        endpoint = "https://api.deepseek.com/v1/chat/completions"
        try:
            if stream:
                response = requests.post(endpoint, headers=headers, json=data, stream=True)
                response.raise_for_status()
                def stream_generator():
                    for line in response.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                line = line[6:]
                            try:
                                json_response = json.loads(line)
                            except Exception as e:
                                print("[Streaming] Could not parse line as JSON:", line)
                                continue
                            # DeepSeek streaming returns 'choices' with 'delta' and 'content'
                            if 'choices' in json_response and json_response['choices']:
                                delta = json_response['choices'][0].get('delta', {})
                                content = delta.get('content')
                                if content:
                                    yield content
                return stream_generator()
            else:
                response = requests.post(endpoint, headers=headers, json=data)
                try:
                    response.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    print("DeepSeek API Error Response:", response.text)
                    raise
                try:
                    result = response.json()
                except Exception as e:
                    print("[Non-streaming] Could not parse response as JSON:", response.text)
                    raise
                # DeepSeek non-streaming returns 'choices' with 'message' and 'content'
                if 'choices' in result and result['choices']:
                    message = result['choices'][0].get('message', {})
                    content = message.get('content')
                    if content:
                        return content
                return "[No valid response from DeepSeek]"
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to DeepSeek: {str(e)}")


    # Streaming handled in get_deepseek_completion; no separate function needed.

    def search_similar_documents(self, query_text):
        # If database is not available, return None to force direct query mode
        if self.db is None:
            return None
        
        try:
            results = self.db.similarity_search_with_relevance_scores(query_text, k=self.max_results)
            if len(results) == 0 or results[0][1] < self.similarity_threshold:
                return None
            return results
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return None

    def format_context(self, results):
        if results is None:
            return ""
        try:
            return "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        except Exception as e:
            print(f"Error formatting context: {str(e)}")
            return ""

    def query(self, query_text, force_direct=False, stream=False):
        try:
            # First attempt to use RAG if database is available and not forced to direct mode
            if not force_direct and self.db is not None:
                try:
                    results = self.search_similar_documents(query_text)
                    if results is not None:
                        context_text = self.format_context(results)
                        rag_prompt = self.PROMPT_TEMPLATE.format(context=context_text, question=query_text)
                        return self.get_deepseek_completion(rag_prompt, stream=stream)
                except Exception as e:
                    print(f"Error in RAG query processing, falling back to direct: {str(e)}")
            
            # Fall back to direct query mode if RAG fails or is not available
            direct_prompt = self.DIRECT_PROMPT_TEMPLATE.format(question=query_text)
            return self.get_deepseek_completion(direct_prompt, stream=stream)
            
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            print(error_message)
            return "I'm sorry, I encountered an error processing your query. Please try again or ask a different question."

if __name__ == "__main__":
    engine = PlantQueryEngine()
    print("PlantQueryEngine initialized and ready to use.")
