from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from llama_parse import LlamaParse
from typing import Dict, Any
import json
import tempfile
import os
from dotenv import load_dotenv
import asyncio
import nest_asyncio
from pathlib import Path

# Apply nest_asyncio to handle nested async loops
nest_asyncio.apply()
load_dotenv()

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; adjust for specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

# JSON Schema definition
TICKET_SCHEMA = {
    "type": "object",
    "properties": {
        "Date of Issue": { "type": "string" },
        "Journey Type": { "type": "string" },
        "From Station": { "type": "string" },
        "To Station": { "type": "string" },
        "Class Value": { "type": "string" },
        "Fare Value": { "type": "number" },
        "Adult/Child Value": { "type": "string" },
        "Validity": { "type": "string" },
        "Timestamp": { "type": "string" },
        "Value": { "type": "string" }
    },
    "required": [
        "Date of Issue",
        "Journey Type",
        "From Station",
        "To Station",
        "Class Value",
        "Fare Value",
        "Adult/Child Value",
        "Validity"
    ]
}

# Initialize LlamaParse
async def get_parser():
    return LlamaParse(
        result_type='structured',
        structured_output=True,
        structured_output_json_schema=json.dumps(TICKET_SCHEMA),
        max_pages=1
    )

def get_file_extension(filename: str) -> str:
    """Extract the file extension from the filename."""
    return Path(filename).suffix.lower()

@app.get("/status")
async def server_status() -> Dict[str, str]:
    """
    Endpoint to check the server status.
    
    Returns:
        Dict[str, str]: A simple status message.
    """
    return {"status": "Server is running"}

@app.post("/parse-ticket")
async def parse_ticket(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Endpoint to parse ticket images using LlamaParse.
    
    Args:
        file (UploadFile): The uploaded image file
        
    Returns:
        Dict[str, Any]: Parsed ticket information
        
    Raises:
        HTTPException: If file upload or processing fails
    """
    try:
        # Get file extension
        file_ext = get_file_extension(file.filename)
        
        # Validate file type
        if not file.content_type.startswith('image/') or file_ext not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file format. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
            )
        
        # Create temporary file with correct extension
        temp_file_path = None
        try:
            # Create a temporary file with the correct extension
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name

            # Initialize parser
            parser = await get_parser()
            
            # Process the image with LlamaParse using async
            documents = await parser.aload_data(temp_file_path)
            
            if not documents or not documents[0].text_resource:
                raise HTTPException(
                    status_code=422,
                    detail="Failed to extract information from the image"
                )
            
            # Extract structured data
            structured_data = json.loads(documents[0].text_resource.text)
            
            return JSONResponse(content=structured_data)
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing image: {str(e)}"
            )
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error handling file upload: {str(e)}"
        )

# Optional: Add error handlers if needed
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
