from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from llama_parse import LlamaParse
from typing import Dict, Any
import json
import tempfile
import os
from dotenv import load_dotenv
import cv2
import numpy as np
from pathlib import Path

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
        "Date of Issue": { 
            "type": "string", 
            "format": "date", 
            "description": "The date the ticket was issued, in YYYY-MM-DD format."
        },
        "Journey Type": { 
            "type": "string", 
            "enum": ["Single", "Return"], 
            "description": "Type of journey the ticket is valid for (Single or Return)."
        },
        "Source Station": { 
            "type": "string", 
            "minLength": 1, 
            "description": "Name of the station where the journey starts."
        },
        "Destination Station": { 
            "type": "string", 
            "minLength": 1, 
            "description": "Name of the station where the journey ends."
        },
        "Class Value": { 
            "type": "string", 
            "enum": ["First Class", "Second Class"], 
            "description": "Class of travel (First Class or Second Class)."
        },
        "Fare Value": { 
            "type": "number", 
            "minimum": 0, 
            "description": "Fare value of the ticket in the applicable currency (e.g., INR)."
        },
        "Adult/Child Value": { 
            "type": "string", 
            "enum": ["Adult", "Child"], 
            "description": "Indicates whether the ticket is for an adult or a child."
        },
        "Validity": { 
            "type": "string", 
            "pattern": "^(\\d+ (Hours|Days))$", 
            "description": "Validity period of the ticket (e.g., '1 Day', '12 Hours')."
        },
        "Timestamp": { 
            "type": "string", 
            "format": "date-time", 
            "description": "Timestamp when the ticket was created, in ISO 8601 format."
        },
        "Value": { 
            "type": "string", 
            "minLength": 1, 
            "description": "Unique ticket number or identifier."
        }
    },
    "required": [
        "Date of Issue",
        "Journey Type",
        "Source Station",
        "Destination Station",
        "Class Value",
        "Fare Value",
        "Adult/Child Value",
        "Validity"
    ]
}

async def get_parser():
    """Initialize LlamaParse with structured output."""
    return LlamaParse(
        result_type='structured',
        structured_output=True,
        structured_output_json_schema=json.dumps(TICKET_SCHEMA),
        max_pages=1
    )

def get_file_extension(filename: str) -> str:
    """Extract the file extension from the filename."""
    return Path(filename).suffix.lower()

def detect_document(image_path: str) -> str:
    """
    Detect and extract the document from the image using edge detection.

    Args:
        image_path (str): Path to the input image.

    Returns:
        str: Path to the extracted document image.

    Raises:
        HTTPException: If document cannot be detected.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur and Canny Edge Detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Loop over contours to find a quadrilateral (document)
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            # Transform the perspective to get a top-down view of the document
            pts = np.array([point[0] for point in approx], dtype="float32")
            rect = cv2.boundingRect(pts)
            cropped = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
            
            # Save the cropped document
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(temp_file.name, cropped)
            return temp_file.name

    raise HTTPException(status_code=422, detail="Document not detected in the image.")

@app.post("/parse-ticket")
async def parse_ticket(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Endpoint to parse ticket images using LlamaParse.

    Args:
        file (UploadFile): The uploaded image file.
        
    Returns:
        Dict[str, Any]: Parsed ticket information.
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
        
        temp_file_path = None
        try:
            # Create a temporary file with the correct extension
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name

            # Detect and extract the document
            extracted_document_path = detect_document(temp_file_path)

            # Initialize LlamaParse
            parser = await get_parser()
            
            # Process the image with LlamaParse
            documents = await parser.aload_data(extracted_document_path)
            
            if not documents or not documents[0].text_resource:
                raise HTTPException(
                    status_code=422,
                    detail="Failed to extract information from the document."
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
            # Clean up temporary files
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            if extracted_document_path and os.path.exists(extracted_document_path):
                os.unlink(extracted_document_path)
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error handling file upload: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
