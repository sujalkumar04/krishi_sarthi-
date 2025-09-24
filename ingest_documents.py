#!/usr/bin/env python3
"""
Document Ingestion Script for Agriculture AI Assistant
Ingests PDF, CSV, and text documents into the FAISS knowledge base
"""

import asyncio
import argparse
import logging
from pathlib import Path
import sys

from app.modules.rag import RAGService
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentIngester:
    """Document ingestion utility"""
    
    def __init__(self):
        self.rag_service = RAGService()
        self.supported_formats = ['.pdf', '.csv', '.txt', '.docx']
    
    async def initialize(self):
        """Initialize the RAG service"""
        await self.rag_service.initialize()
        logger.info("RAG service initialized")
    
    async def ingest_file(self, file_path: Path, doc_type: str = None) -> dict:
        """Ingest a single file"""
        try:
            if not file_path.exists():
                return {"status": "error", "message": f"File not found: {file_path}"}
            
            # Determine document type if not specified
            if not doc_type:
                doc_type = self._get_document_type(file_path)
            
            if doc_type not in ['pdf', 'csv', 'text']:
                return {"status": "error", "message": f"Unsupported document type: {doc_type}"}
            
            logger.info(f"Ingesting {file_path.name} as {doc_type}")
            
            # Create a mock file object for the RAG service
            class MockFile:
                def __init__(self, path):
                    self.path = path
                    self.filename = path.name
                
                async def read(self):
                    with open(self.path, 'rb') as f:
                        return f.read()
            
            mock_file = MockFile(file_path)
            
            # Ingest based on type
            if doc_type == 'pdf':
                result = await self.rag_service.ingest_pdf(mock_file)
            elif doc_type == 'csv':
                result = await self.rag_service.ingest_csv(mock_file)
            else:
                result = await self.rag_service.ingest_text(mock_file)
            
            logger.info(f"Successfully ingested {file_path.name}")
            return result
            
        except Exception as e:
            logger.error(f"Error ingesting {file_path.name}: {e}")
            return {"status": "error", "message": str(e)}
    
    async def ingest_directory(self, directory_path: Path, recursive: bool = False) -> dict:
        """Ingest all supported documents in a directory"""
        try:
            if not directory_path.exists() or not directory_path.is_dir():
                return {"status": "error", "message": f"Directory not found: {directory_path}"}
            
            logger.info(f"Scanning directory: {directory_path}")
            
            # Find all supported files
            files_to_process = []
            if recursive:
                pattern = "**/*"
            else:
                pattern = "*"
            
            for file_path in directory_path.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                    files_to_process.append(file_path)
            
            if not files_to_process:
                return {"status": "warning", "message": "No supported files found"}
            
            logger.info(f"Found {len(files_to_process)} files to process")
            
            # Process files
            results = []
            for file_path in files_to_process:
                result = await self.ingest_file(file_path)
                result['file'] = str(file_path)
                results.append(result)
            
            # Save updated index
            self.rag_service._save_index_and_metadata()
            
            # Summary
            successful = sum(1 for r in results if r['status'] == 'success')
            failed = len(results) - successful
            
            summary = {
                "status": "completed",
                "total_files": len(files_to_process),
                "successful": successful,
                "failed": failed,
                "results": results
            }
            
            logger.info(f"Ingestion completed: {successful} successful, {failed} failed")
            return summary
            
        except Exception as e:
            logger.error(f"Error ingesting directory: {e}")
            return {"status": "error", "message": str(e)}
    
    def _get_document_type(self, file_path: Path) -> str:
        """Determine document type from file extension"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return 'pdf'
        elif suffix == '.csv':
            return 'csv'
        elif suffix in ['.txt', '.md']:
            return 'text'
        elif suffix == '.docx':
            return 'text'  # Will be converted to text
        else:
            return 'text'  # Default to text
    
    def get_index_stats(self) -> dict:
        """Get current index statistics"""
        return self.rag_service.get_index_stats()
    
    def clear_index(self) -> bool:
        """Clear the entire index"""
        return self.rag_service.clear_index()
    
    def export_documents(self, output_path: str) -> bool:
        """Export all documents to JSON"""
        return self.rag_service.export_documents(output_path)

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Ingest documents into Agriculture AI Assistant knowledge base')
    parser.add_argument('path', help='File or directory path to ingest')
    parser.add_argument('--type', '-t', choices=['pdf', 'csv', 'text'], 
                       help='Document type (auto-detected if not specified)')
    parser.add_argument('--recursive', '-r', action='store_true', 
                       help='Recursively scan directories')
    parser.add_argument('--stats', '-s', action='store_true', 
                       help='Show index statistics')
    parser.add_argument('--clear', '-c', action='store_true', 
                       help='Clear the entire index')
    parser.add_argument('--export', '-e', metavar='PATH', 
                       help='Export all documents to JSON file')
    
    args = parser.parse_args()
    
    try:
        # Initialize ingester
        ingester = DocumentIngester()
        await ingester.initialize()
        
        # Handle special operations
        if args.stats:
            stats = ingester.get_index_stats()
            print("\n📊 Knowledge Base Statistics:")
            print("=" * 40)
            for key, value in stats.items():
                print(f"{key:25}: {value}")
            return
        
        if args.clear:
            confirm = input("⚠️  Are you sure you want to clear the entire index? (yes/no): ")
            if confirm.lower() == 'yes':
                success = ingester.clear_index()
                if success:
                    print("✅ Index cleared successfully")
                else:
                    print("❌ Failed to clear index")
            else:
                print("Operation cancelled")
            return
        
        if args.export:
            success = ingester.export_documents(args.export)
            if success:
                print(f"✅ Documents exported to {args.export}")
            else:
                print("❌ Failed to export documents")
            return
        
        # Process path
        path = Path(args.path)
        
        if path.is_file():
            # Ingest single file
            result = await ingester.ingest_file(path, args.type)
            print(f"\n📄 File Ingestion Result:")
            print("=" * 30)
            for key, value in result.items():
                print(f"{key:20}: {value}")
            
            # Save index
            ingester.rag_service._save_index_and_metadata()
            
        elif path.is_dir():
            # Ingest directory
            result = await ingester.ingest_directory(path, args.recursive)
            print(f"\n📁 Directory Ingestion Result:")
            print("=" * 35)
            print(f"Status: {result['status']}")
            print(f"Total Files: {result['total_files']}")
            print(f"Successful: {result['successful']}")
            print(f"Failed: {result['failed']}")
            
            if result['failed'] > 0:
                print(f"\n❌ Failed Files:")
                for r in result['results']:
                    if r['status'] == 'error':
                        print(f"  - {r['file']}: {r['message']}")
            
        else:
            print(f"❌ Path not found: {path}")
            return
        
        # Show updated stats
        print(f"\n📊 Updated Index Statistics:")
        print("=" * 35)
        stats = ingester.get_index_stats()
        for key, value in stats.items():
            print(f"{key:25}: {value}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
