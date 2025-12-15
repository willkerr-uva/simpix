import urllib.request
import os
from PIL import Image
import shutil

def download_image(url, path):
    try:
        print(f"Downloading {url} to {path}...")
        # Add headers to avoid 403 Forbidden on some wikimedia servers
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, path)
        print("Success.")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def resize_image(input_path, output_path, size=(640, 480)):
    try:
        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            return
        img = Image.open(input_path).convert('RGB')
        img = img.resize(size, Image.Resampling.LANCZOS)
        img.save(output_path)
        print(f"Resized {input_path} to {output_path} with size {size}")
    except Exception as e:
        print(f"Error resizing {input_path}: {e}")

if __name__ == "__main__":
    if not os.path.exists("simpix_files"):
        os.makedirs("simpix_files")

    # 1. Download New Images
    monet_url = "https://upload.wikimedia.org/wikipedia/commons/5/54/Claude_Monet%2C_Impression%2C_soleil_levant.jpg"
    renoir_url = "https://upload.wikimedia.org/wikipedia/commons/2/21/Pierre-Auguste_Renoir%2C_Le_Moulin_de_la_Galette.jpg"
    
    download_image(monet_url, "simpix_files/monet_sunrise_orig.jpg")
    download_image(renoir_url, "simpix_files/renoir_moulin_orig.jpg")
    
    # 2. Resize New Images
    resize_image("simpix_files/monet_sunrise_orig.jpg", "simpix_files/monet_sunrise_640.png")
    resize_image("simpix_files/renoir_moulin_orig.jpg", "simpix_files/renoir_moulin_640.png")

    # 3. Prepare Rotunda/Frisbee
    # Assuming they are in the root directory relative to this script execution
    frisbee_src = "frisbe_scott_stadium.png"
    rotunda_src = "rotunda_north_facade.png"
    
    # Copy and ensure size (just in case)
    resize_image(frisbee_src, "simpix_files/frisbee_640.png")
    resize_image(rotunda_src, "simpix_files/rotunda_640.png")
