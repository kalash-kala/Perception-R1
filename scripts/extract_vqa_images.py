import os
import io
import pandas as pd
from PIL import Image
from tqdm import tqdm

def extract_and_save_image(image_dict, image_dir, row_index):
    """
    Extract image bytes from the HF dataset row and save as a .jpg.
    Uses the original filename from the dataset when available,
    falling back to a generated name.

    Returns:
        Absolute path to the saved image file.
    """
    if not isinstance(image_dict, dict):
        return None
        
    image_bytes = image_dict.get("bytes")
    original_name = image_dict.get("path")

    if not image_bytes:
        return None

    if original_name:
        # Use original filename (e.g. COCO_val2014_000000034257.jpg)
        image_filename = os.path.basename(original_name)
    else:
        image_filename = f"vqa_image_{row_index}.jpg"

    # Ensure it's a valid jpeg filename fallback
    if not image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_filename += ".jpg"

    image_path = os.path.join(image_dir, image_filename)

    # Skip if already extracted (idempotent)
    if not os.path.exists(image_path):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(image_path, format="JPEG")
        except Exception as e:
            print(f"Error processing image for row {row_index}: {e}")
            return None

    return image_path

def main():
    input_file = "/home/debarpanb1/kalashkala/visual-question-answering/vqa_stratified_100.parquet"
    output_dir = "/home/debarpanb1/kalashkala/visual-question-answering/processed_for_verl/images"

    print(f"Ensuring output directory exists: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading dataset from {input_file}...")
    try:
        df = pd.read_parquet(input_file)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return
    
    print(f"Total rows to process: {len(df)}")
    print(f"Extracting images...")
    
    extracted_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_dict = row.get('image')
        image_path = extract_and_save_image(image_dict, output_dir, idx)
        if image_path:
            extracted_count += 1
            
    print(f"\nExtraction complete. Successfully processed and saved {extracted_count} images to {output_dir}")

if __name__ == "__main__":
    main()
