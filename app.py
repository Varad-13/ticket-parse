from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
import tempfile
import os
import asyncio
import nest_asyncio
from pathlib import Path
from dotenv import load_dotenv

# Supabase imports
from supabase import create_client, Client

# Razorpay import
import razorpay

# LlamaParse import
from llama_parse import LlamaParse

nest_asyncio.apply()
load_dotenv()

app = FastAPI()

# Enable CORS for all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------
# Supabase Initialization
# ---------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------
# Razorpay Initialization
# ---------------------
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

# JSON Schema definition for parsed tickets
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

# ---------------------
# Ticket Creation Model
# ---------------------
class TicketRequest(BaseModel):
    user_id: str
    from_station: str
    to_station: str
    journey_date: str  # or datetime
    class_value: str
    fare_value: float
    adult_child_value: str
    validity: str
    additional_info: Optional[str] = None

@app.post("/create-ticket")
async def create_ticket(ticket_data: TicketRequest):
    """
    Create a ticket record and a Razorpay order.
    
    Flow:
    1. Create Razorpay order with the ticket fare.
    2. Insert the ticket details + order_id into Supabase.
    3. Return order_id and amount to the frontend.
    """
    try:
        # 1) Create a Razorpay Order
        #    Razorpay expects amount in paise for INR
        amount_in_paise = int(ticket_data.fare_value * 100)

        razorpay_order = razorpay_client.order.create(
            {
                "amount": amount_in_paise,
                "currency": "INR",
                "receipt": f"receipt_{ticket_data.user_id}",
                "payment_capture": 1  # auto-capture
            }
        )

        order_id = razorpay_order["id"]

        # 2) Insert ticket details into Supabase with the order_id
        #    You can customize the table name ("tickets") and data as needed.
        insertion_data = {
            "user_id": ticket_data.user_id,
            "from_station": ticket_data.from_station,
            "to_station": ticket_data.to_station,
            "journey_date": ticket_data.journey_date,
            "class_value": ticket_data.class_value,
            "fare_value": ticket_data.fare_value,
            "adult_child_value": ticket_data.adult_child_value,
            "validity": ticket_data.validity,
            "razorpay_order_id": order_id,
            "payment_status": "CREATED"  # or any custom status
        }

        response = supabase.table("tickets").insert(insertion_data).execute()
        print(response)
        # 3) Return the Razorpay order details to the frontend
        return {
            "order_id": order_id,
            "amount": ticket_data.fare_value,
            "currency": "INR",
            "message": "Ticket created and Razorpay order generated."
        }

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create ticket: {str(e)}"
        )

# Optional: Add error handlers if needed
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Local development entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
