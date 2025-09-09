import pytesseract
from PIL import Image

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image_path = "backend/agents/vision_agents/test_img_3.png"  # Correct relative path from project root

try:
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    print("Extracted Text:")
    print(text)
except Exception as e:
    print(f"Error: {e}")
