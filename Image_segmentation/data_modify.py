import os
import glob

# Set the directory containing your image files
directory_path = 'data/train_masks'

# Use glob to find all .png files in the directory
png_files = glob.glob(os.path.join(directory_path, '*.png'))

# Loop through the list of .png files and delete each one
for file_path in png_files:
    try:
        print("hello")
        os.remove(file_path)
        print(f"Deleted {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

print("Deletion complete.")


# import os
# import glob

# # Set the directory containing your image files
# directory_path = r'data/train_masks'

# # Use glob to find all files ending with _segmentation.png
# files_to_rename = glob.glob(os.path.join(directory_path, '*_segmentation.png'))

# # Loop through the list of files and rename each one
# for file_path in files_to_rename:
#     try:
#         new_file_path = file_path.replace('_segmentation.png', '.png')
#         os.rename(file_path, new_file_path)
#         print(f"Renamed {file_path} to {new_file_path}")
#     except Exception as e:
#         print(f"Error renaming {file_path}: {e}")

# print("Renaming complete.")


# from PIL import Image
# import os
# import glob

# # Set the directory containing your image files
# directory_path = r'data/train_images'

# # Use glob to find all .png files in the directory
# png_files = glob.glob(os.path.join(directory_path, '*.png'))

# for file_path in png_files:
#     try:
#         # Open the PNG image
#         with Image.open(file_path) as img:
#             # Define the new file path with .jpg extension
#             new_file_path = file_path.replace('.png', '.jpg')
#             # Convert and save as .jpg
#             img.convert('RGB').save(new_file_path, 'JPEG')
#             print(f"Converted {file_path} to {new_file_path}")
        
#         # Delete the original .png file
#         os.remove(file_path)
#         print(f"Deleted original file {file_path}")
#     except Exception as e:
#         print(f"Error processing {file_path}: {e}")

# print("Conversion and deletion complete.")

