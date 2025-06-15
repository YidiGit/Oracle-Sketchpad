from PIL import Image
import os

def convert_images(input_dir, output_dir):
    # Check if the output directory exists; if not, create it.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if the output directory exists; if not, create it.
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        # Check if the file is an image file.
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
        
                with Image.open(file_path) as img:
                    
                    img = img.resize((256, 256), Image.LANCZOS)
                    
                    img.info['dpi'] = (300, 300)
                    # Generate the output file path.
                    output_filename = os.path.splitext(filename)[0] + '.jpg'
                    output_file_path = os.path.join(output_dir, output_filename)
                    # Save as JPEG format.
                    img.save(output_file_path, 'JPEG', quality=90)
                    print(f"Converted {filename} to {output_filename}")
            except Exception as e:
                print(f"Error converting {filename}: {e}")

input_directory = 'F:\\十二生肖\\鸡'
output_directory = input_directory+'2'
convert_images(input_directory, output_directory)