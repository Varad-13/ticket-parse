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
import logging

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

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
        "Phone Number": { "type": "number" },
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
            "payment_status": "PAID"  # or any custom status
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

# ---------------------
# Challan Model
# ---------------------
class ChallanRequest(BaseModel):
    user_id: str
    reason: str
    fine_amount: float

@app.post("/issue-challan")
async def issue_challan(challan_data: ChallanRequest):
    """
    Issue a challan for an invalid ticket and generate a Razorpay payment link.
    
    Flow:
    1. Create a Razorpay payment link for the fine amount.
    2. Insert challan details into Supabase.
    3. Return the payment link to the frontend.
    """
    try:
        # 1) Create a Razorpay Payment Link
        payment_link_response = razorpay_client.payment_link.create({
            "amount": int(challan_data.fine_amount * 100),  # Amount in paise
            "currency": "INR",
            "description": f"Challan for Mumbai Local",
            "callback_url": "https://ticket-parse-nextjs.vercel.app/success",  # Update with your frontend's success page
            "callback_method": "get"
        })

        payment_link = payment_link_response["short_url"]
        payment_id = payment_link_response["id"]

        # 2) Insert challan details into Supabase
        insertion_data = {
            "user_id": challan_data.user_id,
            "reason": challan_data.reason,
            "fine_amount": challan_data.fine_amount,
            "razorpay_payment_id": payment_id,
            "payment_status": "PAID"
        }

        response = supabase.table("challans").insert(insertion_data).execute()

        # 3) Return the Razorpay payment link to the frontend
        return {
            "payment_link": payment_link,
            "amount": challan_data.fine_amount,
            "currency": "INR",
            "message": "Challan issued successfully. Use the link to make the payment."
        }

    except Exception as e:
        logger.debug(e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to issue challan: {str(e)}"
        )
    
from fastapi import Request

@app.post("/verify-payment")
async def verify_payment(request: Request):
    try:
        body = await request.json()
        payment_id = body.get("razorpay_payment_id")
        payment_link = body.get("razorpay_payment_link_id")
        order_id = body.get("razorpay_order_id")
        signature = body.get("razorpay_signature")
        logger.debug(body)

        if order_id:
            try:
                status = razorpay_client.utility.verify_payment_signature({
                    "razorpay_order_id": order_id,
                    "razorpay_payment_id": payment_id,
                    "razorpay_signature": signature
                })

            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Signature verification failed: {str(e)}")
            # Fetch payment status from Razorpay
            payment_details = razorpay_client.payment.fetch(payment_id)
            status = payment_details.get("status")
            if status != "captured":
                raise HTTPException(status_code=400, detail="Payment not captured yet")
        else:
            challan_response = supabase.table("challans").select("id, payment_status").eq("razorpay_payment_id", payment_link).execute()
        
        return {"message": "Payment verified and updated successfully", "status": "PAID"}

    except Exception as e:
        logger.exception("Error verifying payment")
        raise HTTPException(status_code=500, detail=f"Error verifying payment: {str(e)}")

@app.get("/tickets/{user_id}")
async def get_paid_tickets(user_id: str):
    """
    Get all tickets with status "PAID" for a given user_id.

    Args:
        user_id (str): The ID of the user whose paid tickets need to be fetched.

    Returns:
        Dict[str, Any]: List of paid tickets for the user.
    
    Raises:
        HTTPException: If no tickets are found or an error occurs.
    """
    try:
        # Query Supabase for tickets where user_id matches and payment_status is "PAID"
        response = (
            supabase.table("tickets")
            .select("*")
            .eq("user_id", user_id)
            .eq("payment_status", "PAID")  # Filter for PAID status
            .execute()
        )

        if not response.data:
            raise HTTPException(status_code=404, detail="No paid tickets found for this user.")

        return {"tickets": response.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching tickets: {str(e)}")


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
