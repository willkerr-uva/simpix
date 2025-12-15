from PIL import Image
import os

def resize_image(input_path, output_path, size=(640, 480)):
    try:
        img = Image.open(input_path)
        img = img.resize(size, Image.Resampling.LANCZOS)
        # Convert to RGB to ensure 3 channels (remove alpha if present)
        img = img.convert('RGB')
        img.save(output_path)
        print(f"Resized {input_path} to {output_path} with size {size}")
    except Exception as e:
        print(f"Error resizing {input_path}: {e}")

if __name__ == "__main__":
    if not os.path.exists("simpix_files"):
        os.makedirs("simpix_files")
        
    # Source images from the cloned repo
    monet1 = "c:/Users/willi/projects/simpix/simpix/Images/Claude_Monet_Le_Grand_Canal.png"
    monet2 = "c:/Users/willi/projects/simpix/simpix/Images/Claude_Monet_The_Cliffs_at_Etretat.png"
    
    resize_image(monet1, "simpix_files/monet_canal_640x480.png")
    resize_image(monet2, "simpix_files/monet_cliffs_640x480.png")
    
    # Also resize a very small version for fast testing
    resize_image(monet1, "simpix_files/test_src.png", size=(40, 30))
    resize_image(monet2, "simpix_files/test_tgt.png", size=(40, 30))
