import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config

cfg = load_config()

# Client vLLM server
client = OpenAI(
    api_key = "EMPTY",
    base_url = "http://localhost:8000/v1"
)


# encode image to b64
def enc_b64(image, max_size=(512, 512)):

    image = image.copy()
    image.thumbnail(max_size, Image.LANCZOS)  # resize to fit in vram
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return b64


def generate_answer(query, images):
    
    """Sends the query and the retrieved images to the gen VLM"""
    
    # Prep context 
    messages = [
        {"role": "system", "content": "You are a precise data extraction assistant. First, describe the layout of the page. Second, locate the specific figure requested. Third, describe exactly what you see in that figure step-by-step before answering the user's specific question."}
    ]
    
    user_content = [{"type": "text", "text": query}]
    
    # Add top k images to prompt
    for img in images:
        b64_img = enc_b64(img)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
        })
        
    messages.append({"role": "user", "content": user_content})


    # call VLM
    try:
        response = client.chat.completions.create(
            model=cfg.models.vlm_model,
            messages=messages,
            temperature=0.05, 
            max_tokens=1000
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"VLM Error: {str(e)}"

# Test vlm
if __name__ == "__main__":
    test_img = Image.new('RGB', (100, 100), color = 'red')
    print(generate_answer("What color is this image?", [test_img]))