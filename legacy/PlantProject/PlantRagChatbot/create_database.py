"""
Create a plant database with plant types as metadata.
This script extracts plant types from filenames, loads PDF content, chunks the text,
and ensures each chunk maintains the plant type metadata.
Uses HuggingFace embeddings for better compatibility.
"""
import os
import sys
import glob
import time
import shutil
import requests
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Configure paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(SCRIPT_DIR, "chroma")
DATA_PATH = os.path.join(SCRIPT_DIR, "data")

def extract_plant_type(filename):
    """Extract plant type from filename using keywords."""
    # List of known plant types to search for
    plant_types = [
        "apple", "tomato", "potato", "corn", "bean", "soy", "wheat", 
        "rice", "barley", "oat", "grape", "orange", "lemon", "peach",
        "pear", "cherry", "strawberry", "blueberry", "raspberry",
        "blackberry", "pepper", "carrot", "cucumber", "lettuce"
    ]
    
    # Convert filename to lowercase for case-insensitive matching
    base_name = os.path.basename(filename).lower()
    
    # Special case for general information documents
    if "general" in base_name or "guide" in base_name or "info" in base_name:
        return "general"
    
    # Special case for blueberries (plurals)
    if "blueberries" in base_name:
        return "blueberry"
        
    # Special case for other plurals
    for plant in plant_types:
        plural = plant + "s"
        if plural in base_name:
            return plant
    
    # Try to find a plant type in the filename
    for plant in plant_types:
        if plant in base_name:
            return plant
    
    # Return unknown if no plant type found
    return "unknown"

def check_ollama_status():
    """Check if Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434/api/version")
        if response.status_code == 200:
            version = response.json().get("version", "unknown")
            print(f"Ollama server is running: {version}")
            return True
        else:
            print("Ollama server returned unexpected status code:", response.status_code)
            return False
    except Exception as e:
        print(f"Failed to connect to Ollama server: {str(e)}")
        return False

def create_database():
    """Create the Chroma database with plant metadata."""
    print("\n===== Creating Plant Database =====\n")
    
    # 1. Scan files and extract plant types
    print("Scanning data directory for files...")
    file_paths = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    file_count = len(file_paths)
    print(f"Found {file_count} files")
    
    # Track plant types
    plant_counts = {}
    unknown_files = []
    
    # Map files to plant types
    file_plant_mappings = {}
    for file_path in file_paths:
        plant_type = extract_plant_type(file_path)
        file_plant_mappings[file_path] = plant_type
        
        # Track unknown files
        if plant_type == "unknown":
            unknown_files.append(os.path.basename(file_path))
            
        # Update plant count
        if plant_type in plant_counts:
            plant_counts[plant_type] += 1
        else:
            plant_counts[plant_type] = 1
    
    # Print unknown files
    if unknown_files:
        print("\nFiles with UNKNOWN plant type (consider renaming these):")
        for filename in unknown_files:
            print(f"  - {filename}")
    
    # 2. Load documents and split into chunks
    print("\nLoading and chunking documents...")
    all_chunks = []
    chunk_counts = {}
    
    # Text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    # Process each PDF file
    for file_path in file_paths:
        try:
            # Get plant type from our mapping
            plant_type = file_plant_mappings[file_path]
            
            # Load the PDF - using custom loading to handle errors
            documents = []
            try:
                # First, try the standard loader
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                
                if not documents or len(documents) == 0:
                    # If standard loader failed, try a more direct approach
                    print(f"    Warning: Standard loader extracted 0 pages from {os.path.basename(file_path)}")
                    # This would be where we'd try an alternative extraction method
            except Exception as pdf_error:
                print(f"    Warning: Error reading PDF {os.path.basename(file_path)}: {str(pdf_error)}")
                
            # If we still couldn't extract any text, create a simple document with filename
            if not documents or len(documents) == 0:
                print(f"    Creating fallback document for {os.path.basename(file_path)}")
                documents = [Document(
                    page_content=f"Information about {plant_type} plants from {os.path.basename(file_path)}",
                    metadata={"source": file_path}
                )]
            
            # Add plant type to document metadata
            for doc in documents:
                doc.metadata["plant_type"] = plant_type
            
            # Split documents into chunks
            chunks = text_splitter.split_documents(documents)
            
            # Track chunks per file
            file_chunk_count = len(chunks)
            print(f"  Processed {os.path.basename(file_path)}: {file_chunk_count} chunks (plant: {plant_type})")
            
            # Add chunks to collection
            all_chunks.extend(chunks)
            
            # Update chunk counts per plant type
            if plant_type in chunk_counts:
                chunk_counts[plant_type] += file_chunk_count
            else:
                chunk_counts[plant_type] = file_chunk_count
                
        except Exception as e:
            print(f"  Error processing {file_path}: {str(e)}")
    
    # Report on chunks
    print(f"\nCreated {len(all_chunks)} total chunks from {file_count} files")
    
    # Print plant type distribution by chunks
    print("\nPlant type distribution in chunks:")
    for plant, count in sorted(chunk_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {plant}: {count} chunks")
    
    # 3. Check Ollama status (even though we're using HuggingFace)
    print("\nChecking Ollama server status...")
    check_ollama_status()
    
    # 4. Set up HuggingFace embeddings
    print("\nConfiguring HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # 5. Remove existing database if it exists
    if os.path.exists(CHROMA_PATH):
        print(f"Removing existing database at {CHROMA_PATH}")
        try:
            # Force removal of the directory
            shutil.rmtree(CHROMA_PATH, ignore_errors=True)
            print("Successfully removed existing database directory")
        except Exception as e:
            print(f"Error removing directory: {str(e)}")
    
    # 6. Create the database
    print("\nCreating embeddings and storing in Chroma database...")
    try:
        if not all_chunks:
            print("ERROR: No chunks created. Cannot create database.")
            return False
            
        # Create Chroma database with chunks
        db = Chroma.from_documents(
            documents=all_chunks, 
            embedding=embeddings, 
            persist_directory=CHROMA_PATH
        )
        
        # Test the database
        print("\nTesting database with similarity search...")
        # Use the content of the first chunk for the test search
        test_text = all_chunks[0].page_content[:100]  # First 100 chars of first chunk
        test_results = db.similarity_search(test_text, k=1)
        
        if test_results and len(test_results) > 0:
            print(f"Database creation successful - found {len(test_results)} results")
            print(f"Sample chunk plant type: {test_results[0].metadata.get('plant_type', 'unknown')}")
            print(f"Total chunks in database: {len(all_chunks)}")
        else:
            print("Database creation may have issues - no search results returned")
        
        print(f"\nDatabase saved to: {CHROMA_PATH}")
        
    except Exception as e:
        print(f"\nError creating database: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False
    
    return True

if __name__ == "__main__":
    print(f"Chroma database will be created at: {CHROMA_PATH}")
    print(f"Loading data from: {DATA_PATH}")
    
    success = create_database()
    
    print("\n===== Database Creation Complete =====")
    if success:
        print("\nYou can now run create_embedding_notebook.py to visualize the data")
