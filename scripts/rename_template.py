import os
import sys

def replace_in_file(file_path, old_str, new_str):
    """Replace all occurrences of old_str with new_str in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        if old_str in content:
            new_content = content.replace(old_str, new_str)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)
            print(f"Updated content in file: {file_path}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def rename_dirs_and_files(base_dir, old_str, new_str):
    """Rename directories and files containing old_str."""
    for root, dirs, files in os.walk(base_dir, topdown=False):
        # Rename directories
        for dir_name in dirs:
            if old_str in dir_name:
                old_dir_path = os.path.join(root, dir_name)
                new_dir_path = os.path.join(root, dir_name.replace(old_str, new_str))
                os.rename(old_dir_path, new_dir_path)
                print(f"Renamed directory: {old_dir_path} -> {new_dir_path}")

        # Rename files
        for file_name in files:
            if old_str in file_name:
                old_file_path = os.path.join(root, file_name)
                new_file_path = os.path.join(root, file_name.replace(old_str, new_str))
                os.rename(old_file_path, new_file_path)
                print(f"Renamed file: {old_file_path} -> {new_file_path}")

def process_directory(base_dir, old_str, new_str):
    """Process the directory to replace old_str with new_str in file contents, names, and paths."""
    # Replace in file contents
    for root, dirs, files in os.walk(base_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name.endswith(('.py', '.yaml', '.md', '.toml')):
                replace_in_file(file_path, old_str, new_str)

    # Rename files and directories
    rename_dirs_and_files(base_dir, old_str, new_str)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_template.py <new_name>")
        sys.exit(1)

    new_name = sys.argv[1]
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
    base_directory = os.path.abspath(os.path.join(script_dir, ".."))  # Assume script is in 'scripts' folder
    old_name = "isaac_lab_from_scrach"

    print(f"Replacing '{old_name}' with '{new_name}' in '{base_directory}'...")
    process_directory(base_directory, old_name, new_name)
    print("Replacement complete!")
