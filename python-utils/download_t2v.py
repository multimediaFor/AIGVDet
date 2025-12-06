import gdown
import zipfile
import os

# Google Drive file ID
file_id = "1FT06IRiy1oB1jHWBEarUI99DFCk6VHxf"

# Output path for the downloaded file
output_path = "downloaded_file.zip"

# Download the file from Google Drive
gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# Check if the file was downloaded
if os.path.exists(output_path):
    print(f"File downloaded successfully: {output_path}")

    # Unzip the file
    unzip_dir = "unzipped_files"
    if not os.path.exists(unzip_dir):
        os.makedirs(unzip_dir)

    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)
        print(f"Files extracted to: {unzip_dir}")

    # Optionally, you can delete the zip file after extraction
    os.remove(output_path)
    print("Zip file removed after extraction.")
else:
    print("Download failed.")
