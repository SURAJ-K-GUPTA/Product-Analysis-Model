from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import openpyxl
import re
import os
import torch
import gc
from io import BytesIO
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import multiprocessing
import signal

# Cleanup semaphore resources (to prevent leaked semaphore warnings)
def cleanup_semaphores():
    try:
        multiprocessing.resource_tracker._semaphore_tracker._cache.clear()
    except AttributeError:
        pass
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

cleanup_semaphores()

# Initialize FastAPI app
app = FastAPI()

# Mount the frontend directory as static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Load the model and processor during startup
@app.on_event("startup")
def load_model_and_initialize():
    global model, processor
    try:
        # Load model to GPU if available
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype="auto",
            device_map="auto",  # Automatically use GPU if available
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError("Model loading failed. Check your configuration.")

    # Ensure Excel file exists
    file_path = "product_analysis.xlsx"
    if not os.path.exists(file_path):
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Product Analysis"
        headers = ["Product Name", "Category", "Quantity", "Count", "Expiry Date", "Freshness Index", "Shelf Life"]
        sheet.append(headers)
        workbook.save(file_path)

# Regex patterns for analysis
packaged_product_pattern = r"Product Name: (.*)\n  - Product Category: (.*)\n  - Product Quantity: (.*)\n  - Product Count: (.*)\n  - Expiry Date: (.*)"
fruits_vegetables_pattern = r"Type of fruit/vegetable: (.*)\n  - Freshness Index: (.*)\n  - Estimated Shelf Life: (.*)"

@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    try:
        with open("frontend/index.html", "r") as f:
            return f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading homepage: {e}")

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):
    print("image route")
    try:
        # Log file metadata
        print(f"File name: {file.filename}")
        print(f"Content type: {file.content_type}")

        # Validate the uploaded file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG image.")

        # Read and process the uploaded image
        contents = await file.read()  # Read the file contents into memory
        print(f"File content (first 100 bytes): {contents[:100]}")  # Debug raw content

        try:
            image = Image.open(BytesIO(contents))  # Open image with Pillow
            image = image.resize((512, 512))
        except Exception as e:
            print(f"Pillow error: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

        # Proceed with the rest of the logic
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image_url": "Captured from webcam"},
                    {"type": "text", 
                     "text": """This image contains fruits, vegetables, or packaged products.
                        Please analyze the image and provide:
                        - For packaged products:
                            - Product Name
                            - Product Category
                            - Product Quantity
                            - Product Count
                            - Expiry Date (if available)
                        - For fruits and vegetables:
                            - Type of fruit/vegetable
                            - Freshness Index (based on visual cues)
                            - Estimated Shelf Life"""
                            }
                ]
            }
        ]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        # Move inputs to the same device as the model
        model_device = next(model.parameters()).device
        inputs = inputs.to(model_device)

        # Generate model output
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=1024)
        output_text = processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        # Update Excel
        file_path = "product_analysis.xlsx"
        workbook = openpyxl.load_workbook(file_path)
        sheet = workbook.active
        sheet.append([output_text])
        workbook.save(file_path)

        return {"message": "Analysis completed successfully", "output": output_text}

    except Exception as e:
        print(f"Backend error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/download-excel/")
async def download_excel():
    file_path = "product_analysis.xlsx"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Excel file not found.")
    return FileResponse(
        file_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="product_analysis.xlsx",
    )
