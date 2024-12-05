from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import openpyxl
import re
import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Initialize FastAPI app
app = FastAPI()

# Mount the frontend directory as static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Load the Qwen2 model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Ensure the Excel file exists at startup
@app.on_event("startup")
def ensure_excel_file_exists():
    file_path = "product_analysis.xlsx"
    if not os.path.exists(file_path):
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Product Analysis"
        headers = ["Product Name", "Category", "Quantity", "Count", "Expiry Date", "Freshness Index", "Shelf Life"]
        sheet.append(headers)
        workbook.save(file_path)


# Regular expression patterns
packaged_product_pattern = r"Product Name: (.*)\n  - Product Category: (.*)\n  - Product Quantity: (.*)\n  - Product Count: (.*)\n  - Expiry Date: (.*)"
fruits_vegetables_pattern = r"Type of fruit/vegetable: (.*)\n  - Freshness Index: (.*)\n  - Estimated Shelf Life: (.*)"


# Route to serve the homepage
@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    with open("frontend/index.html", "r") as f:
        return f.read()


# API endpoint to analyze uploaded images
@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Open the uploaded image
        image = Image.open(file.file)
        image = image.resize((512, 512))

        # Prepare the text prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image_url": "Captured from webcam"
                    },
                    {
                        "type": "text",
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

        # Prepare the input for the model
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate output from the model
        output_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        # Extract data using regex
        packaged_product_match = re.search(packaged_product_pattern, output_text)
        fruits_vegetables_match = re.search(fruits_vegetables_pattern, output_text)

        if packaged_product_match:
            product_name = packaged_product_match.group(1).strip()
            category = packaged_product_match.group(2).strip()
            quantity = packaged_product_match.group(3).strip()
            count = packaged_product_match.group(4).strip()
            expiry_date = packaged_product_match.group(5).strip()
        else:
            product_name = category = quantity = count = expiry_date = "N/A"

        if fruits_vegetables_match:
            product_name = fruits_vegetables_match.group(1).strip()
            category = "Fruit/Vegetable"
            freshness_index = fruits_vegetables_match.group(2).strip()
            shelf_life = fruits_vegetables_match.group(3).strip()
        else:
            freshness_index = shelf_life = "N/A"

        # Append to the Excel sheet
        file_path = "product_analysis.xlsx"
        if os.path.exists(file_path):
            workbook = openpyxl.load_workbook(file_path)
            sheet = workbook.active
        else:
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            headers = ["Product Name", "Category", "Quantity", "Count", "Expiry Date", "Freshness Index", "Shelf Life"]
            sheet.append(headers)

        sheet.append([product_name, category, quantity, count, expiry_date, freshness_index, shelf_life])
        workbook.save(file_path)

        return {"message": "Analysis completed successfully", "output": output_text}
    except Exception as e:
        return {"error": str(e)}


# API endpoint to download the Excel file
@app.get("/download-excel/")
async def download_excel():
    file_path = "product_analysis.xlsx"
    if not os.path.exists(file_path):
        raise RuntimeError(f"File at path {file_path} does not exist.")
    return FileResponse(
        file_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="product_analysis.xlsx",
    )
