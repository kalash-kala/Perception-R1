import os
import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

def get_api_key(env_path: str) -> str:
    """
    Extracts Gemini API Key from .env file, handling the specific format:
    Gemini-API-Key = AIza...
    """
    if not os.path.exists(env_path):
        # Try generic relative search if not found
        script_dir = Path(__file__).resolve().parent
        root_env = script_dir.parent / ".env"
        if root_env.exists():
            env_path = str(root_env)
        else:
            raise FileNotFoundError(f".env file not found at {env_path}")
    
    # Try generic load_dotenv first
    load_dotenv(env_path)
    # The user's .env uses 'Gemini-API-Key' which load_dotenv might not like if it contains dashes
    # but we'll check common names anyway
    api_key = os.getenv("Gemini-API-Key") or os.getenv("GEMINI_API_KEY")
    
    if api_key:
        return api_key

    # Fallback to manual parsing if load_dotenv fails with the custom format
    with open(env_path, "r") as f:
        for line in f:
            if "Gemini" in line and "=" in line:
                key, val = line.split("=", 1)
                if "Key" in key:
                    return val.strip()
    
    raise ValueError("Gemini-API-Key not found in .env file")

def setup_gemini(api_key: str, model_name: str = "gemini-1.5-flash"):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def annotate_image(model, image_path: Path, prompt: str) -> str:
    try:
        image = Image.open(image_path)
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Annotate images using Gemini API")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images to annotate")
    parser.add_argument("--output_file", type=str, default="annotations.jsonl", help="File to save annotations (jsonl format)")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.", help="Prompt for annotation")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash", help="Gemini model version")
    parser.add_argument("--extensions", nargs="+", default=[".jpg", ".jpeg", ".png", ".webp"], help="Image extensions to process")
    parser.add_argument("--env_path", type=str, default=None, help="Path to .env file (optional, defaults to searching Perception-R1/.env)")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of images to process")
    
    args = parser.parse_args()

    # Search for .env if not provided
    env_path = args.env_path
    if env_path is None:
        script_dir = Path(__file__).resolve().parent
        root_env = script_dir.parent / ".env"
        env_path = str(root_env) if root_env.exists() else ".env"

    try:
        api_key = get_api_key(env_path)
    except Exception as e:
        print(f"Error loading API Key: {e}")
        return

    model = setup_gemini(api_key, args.model)
    image_dir = Path(args.image_dir)
    image_files = [f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in args.extensions]
    
    if args.limit:
        image_files = image_files[:args.limit]

    if not image_files:
        print(f"No images with extensions {args.extensions} found in {image_dir}")
        return

    print(f"Found {len(image_files)} images in {image_dir}")
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process images and append to results list for safety
    with open(output_path, "a", encoding="utf-8") as f:
        for img_path in tqdm(image_files, desc="Annotating"):
            annotation = annotate_image(model, img_path, args.prompt)
            
            result = {
                "image_id": img_path.name,
                "image_path": str(img_path.absolute()),
                "annotation": annotation,
                "timestamp": time.time()
            }
            
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            
            # Simple rate limiting for free-tier users if needed
            # time.sleep(1)

    print(f"Annotations saved to {output_path}")

if __name__ == "__main__":
    main()
