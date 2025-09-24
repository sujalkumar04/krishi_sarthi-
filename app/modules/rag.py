import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import hashlib

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import pandas as pd

from config import config

logger = logging.getLogger(__name__)

class RAGService:
    """Retrieval-Augmented Generation service using FAISS for vector storage"""
    
    def __init__(self):
        self.vector_db_path = config.VECTOR_DB_PATH
        self.knowledge_base_path = config.KNOWLEDGE_BASE_PATH
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
        self.top_k = config.TOP_K_RETRIEVAL
        
        # Initialize components
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.document_metadata = {}
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the RAG service"""
        try:
            # Load embedding model
            self.embedding_model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
            logger.info(f"Loaded embedding model: {config.LOCAL_EMBEDDING_MODEL}")
            
            # Load or create FAISS index
            self._load_or_create_index()
            
            # Load existing documents
            self._load_document_metadata()
            
            self.is_initialized = True
            logger.info("RAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG service: {e}")
            self.is_initialized = False
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create a new one"""
        try:
            index_path = Path(self.vector_db_path) / "faiss_index.bin"
            metadata_path = Path(self.vector_db_path) / "metadata.json"
            
            if index_path.exists() and metadata_path.exists():
                # Load existing index
                self.index = faiss.read_index(str(index_path))
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                logger.info(f"Loaded existing FAISS index with {len(self.documents)} documents")
            else:
                # Create new index
                dimension = self.embedding_model.get_sentence_embedding_dimension()
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                self.documents = []
                logger.info(f"Created new FAISS index with dimension {dimension}")
                
        except Exception as e:
            logger.error(f"Error loading/creating FAISS index: {e}")
            # Create fallback index
            dimension = 384  # Default dimension for all-MiniLM-L6-v2
            self.index = faiss.IndexFlatIP(dimension)
            self.documents = []
    
    def _load_document_metadata(self):
        """Load document metadata from storage"""
        try:
            metadata_path = Path(self.vector_db_path) / "document_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.document_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.document_metadata)} documents")
            else:
                self.document_metadata = {}
                
        except Exception as e:
            logger.error(f"Error loading document metadata: {e}")
            self.document_metadata = {}
    
    def _save_index_and_metadata(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Create directory if it doesn't exist
            Path(self.vector_db_path).mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            index_path = Path(self.vector_db_path) / "faiss_index.bin"
            faiss.write_index(self.index, str(index_path))
            
            # Save documents
            metadata_path = Path(self.vector_db_path) / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            
            # Save document metadata
            doc_metadata_path = Path(self.vector_db_path) / "document_metadata.json"
            with open(doc_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.document_metadata, f, ensure_ascii=False, indent=2)
                
            logger.info("FAISS index and metadata saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving index and metadata: {e}")
    
    async def ingest_pdf(self, file) -> Dict[str, Any]:
        """Ingest PDF document into the knowledge base"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Read PDF content
            content = await self._read_pdf_content(file)
            
            # Process and chunk the content
            chunks = self._create_chunks(content)
            
            # Generate embeddings and add to index
            documents_added = await self._add_chunks_to_index(chunks, file.filename, "pdf")
            
            # Save updated index
            self._save_index_and_metadata()
            
            return {
                "documents_added": 1,
                "chunks_created": len(chunks),
                "filename": file.filename,
                "content_length": len(content),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error ingesting PDF: {e}")
            return {
                "documents_added": 0,
                "chunks_created": 0,
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            }
    
    async def ingest_csv(self, file) -> Dict[str, Any]:
        """Ingest CSV document into the knowledge base"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Read CSV content
            content = await self._read_csv_content(file)
            
            # Process and chunk the content
            chunks = self._create_chunks(content)
            
            # Generate embeddings and add to index
            documents_added = await self._add_chunks_to_index(chunks, file.filename, "csv")
            
            # Save updated index
            self._save_index_and_metadata()
            
            return {
                "documents_added": 1,
                "chunks_created": len(chunks),
                "filename": file.filename,
                "content_length": len(content),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error ingesting CSV: {e}")
            return {
                "documents_added": 0,
                "chunks_created": 0,
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            }
    
    async def ingest_text(self, file) -> Dict[str, Any]:
        """Ingest text document into the knowledge base"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Read text content
            content = await self._read_text_content(file)
            
            # Process and chunk the content
            chunks = self._create_chunks(content)
            
            # Generate embeddings and add to index
            documents_added = await self._add_chunks_to_index(chunks, file.filename, "text")
            
            # Save updated index
            self._save_index_and_metadata()
            
            return {
                "documents_added": 1,
                "chunks_created": len(chunks),
                "filename": file.filename,
                "content_length": len(content),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error ingesting text: {e}")
            return {
                "documents_added": 0,
                "chunks_created": 0,
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            }
    
    async def _read_pdf_content(self, file) -> str:
        """Read content from PDF file"""
        try:
            content = await file.read()
            
            # Create temporary file for PyPDF2
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # Extract text from PDF
            text_content = ""
            with open(temp_file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return text_content.strip()
            
        except Exception as e:
            logger.error(f"Error reading PDF content: {e}")
            raise Exception(f"Failed to read PDF content: {str(e)}")
    
    async def _read_csv_content(self, file) -> str:
        """Read content from CSV file"""
        try:
            content = await file.read()
            
            # Parse CSV and convert to text
            import io
            csv_text = content.decode('utf-8')
            df = pd.read_csv(io.StringIO(csv_text))
            
            # Convert DataFrame to text representation
            text_content = df.to_string(index=False)
            
            return text_content
            
        except Exception as e:
            logger.error(f"Error reading CSV content: {e}")
            raise Exception(f"Failed to read CSV content: {str(e)}")
    
    async def _read_text_content(self, file) -> str:
        """Read content from text file"""
        try:
            content = await file.read()
            return content.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error reading text content: {e}")
            raise Exception(f"Failed to read text content: {str(e)}")
    
    def _create_chunks(self, text: str) -> List[str]:
        """Create text chunks with overlap"""
        try:
            chunks = []
            words = text.split()
            
            if len(words) <= self.chunk_size:
                chunks.append(text)
                return chunks
            
            # Create overlapping chunks
            for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = " ".join(chunk_words)
                chunks.append(chunk_text)
                
                # Stop if we've covered the entire text
                if i + self.chunk_size >= len(words):
                    break
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            return [text]
    
    async def _add_chunks_to_index(self, chunks: List[str], filename: str, doc_type: str) -> int:
        """Add document chunks to the FAISS index"""
        try:
            documents_added = 0
            
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.embedding_model.encode([chunk])[0]
                
                # Add to FAISS index
                self.index.add(np.array([embedding], dtype=np.float32))
                
                # Store document metadata
                doc_id = f"{filename}_{i}"
                self.documents.append({
                    "id": doc_id,
                    "content": chunk,
                    "filename": filename,
                    "doc_type": doc_type,
                    "chunk_index": i,
                    "embedding_dimension": len(embedding),
                    "added_at": datetime.now().isoformat()
                })
                
                # Store additional metadata
                self.document_metadata[doc_id] = {
                    "filename": filename,
                    "doc_type": doc_type,
                    "chunk_index": i,
                    "content_length": len(chunk),
                    "added_at": datetime.now().isoformat(),
                    "content_hash": hashlib.md5(chunk.encode()).hexdigest()
                }
                
                documents_added += 1
            
            logger.info(f"Added {documents_added} chunks from {filename} to index")
            return documents_added
            
        except Exception as e:
            logger.error(f"Error adding chunks to index: {e}")
            return 0
    
    async def retrieve_relevant_docs(self, query: str, intent_result: Dict[str, Any], 
                                   top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query
            intent_result: Intent and entity extraction result
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with scores
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            if not self.documents or self.index.ntotal == 0:
                logger.warning("No documents in index for retrieval")
                return []
            
            # Use provided top_k or default
            k = top_k if top_k is not None else self.top_k
            k = min(k, len(self.documents))
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search in FAISS index
            scores, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32), k
            )
            
            # Retrieve documents
            retrieved_docs = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc["similarity_score"] = float(score)
                    doc["rank"] = len(retrieved_docs) + 1
                    retrieved_docs.append(doc)
            
            # Sort by score (descending)
            retrieved_docs.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}...")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    async def search_by_intent(self, intent: str, entities: Dict[str, Any], 
                              top_k: int = None) -> List[Dict[str, Any]]:
        """Search for documents relevant to a specific intent and entities"""
        try:
            # Create search query based on intent and entities
            search_terms = [intent]
            
            # Add entity terms
            for entity_type, entity_values in entities.items():
                if isinstance(entity_values, list):
                    search_terms.extend(entity_values)
                else:
                    search_terms.append(str(entity_values))
            
            # Combine search terms
            search_query = " ".join(search_terms)
            
            # Retrieve documents
            return await self.retrieve_relevant_docs(search_query, {"intent": intent, "entities": entities}, top_k)
            
        except Exception as e:
            logger.error(f"Error searching by intent: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the FAISS index"""
        try:
            if not self.is_initialized:
                return {"status": "not_initialized"}
            
            return {
                "total_documents": len(self.documents),
                "index_size": self.index.ntotal if self.index else 0,
                "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension() if self.embedding_model else 0,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "document_types": list(set(doc.get("doc_type", "unknown") for doc in self.documents)),
                "last_updated": max((doc.get("added_at", "1970-01-01") for doc in self.documents), default="unknown")
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def clear_index(self) -> bool:
        """Clear the entire FAISS index"""
        try:
            if self.index:
                self.index.reset()
            self.documents = []
            self.document_metadata = {}
            
            # Save empty index
            self._save_index_and_metadata()
            
            logger.info("FAISS index cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            return False
    
    def remove_document(self, filename: str) -> bool:
        """Remove a specific document from the index"""
        try:
            if not self.is_initialized:
                return False
            
            # Find documents to remove
            docs_to_remove = [i for i, doc in enumerate(self.documents) if doc["filename"] == filename]
            
            if not docs_to_remove:
                logger.warning(f"Document {filename} not found in index")
                return False
            
            # Remove from documents list (in reverse order to maintain indices)
            for i in reversed(docs_to_remove):
                doc_id = self.documents[i]["id"]
                del self.documents[i]
                if doc_id in self.document_metadata:
                    del self.document_metadata[doc_id]
            
            # Rebuild index
            self._rebuild_index()
            
            logger.info(f"Removed document {filename} from index")
            return True
            
        except Exception as e:
            logger.error(f"Error removing document {filename}: {e}")
            return False
    
    def _rebuild_index(self):
        """Rebuild the FAISS index from documents"""
        try:
            if not self.documents:
                self.index.reset()
                return
            
            # Get embedding dimension
            dimension = self.embedding_model.get_sentence_embedding_dimension()
            
            # Create new index
            new_index = faiss.IndexFlatIP(dimension)
            
            # Add all documents
            for doc in self.documents:
                embedding = self.embedding_model.encode([doc["content"]])[0]
                new_index.add(np.array([embedding], dtype=np.float32))
            
            # Replace old index
            self.index = new_index
            
            logger.info("FAISS index rebuilt successfully")
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
    
    def export_documents(self, output_path: str) -> bool:
        """Export all documents to a JSON file"""
        try:
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "total_documents": len(self.documents),
                "documents": self.documents,
                "metadata": self.document_metadata
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Exported {len(self.documents)} documents to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting documents: {e}")
            return False
