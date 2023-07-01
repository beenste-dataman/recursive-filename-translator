import os
import shutil
import torch
import argparse
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

def translate(text: str, model, tokenizer):
    """Translates the text using the MarianMT model."""

    try:
        # Tokenize the text
        tokenized_text = tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')

        # Translate the text
        translated_text = model.generate(**tokenized_text)

        # Decode the translated text
        decoded_text = tokenizer.batch_decode(translated_text, skip_special_tokens=True)[0]

        return decoded_text
    except Exception as e:
        print(f"Error during translation: {e}")
        return text  # If there's an error, return the original text


def translate_filenames(source_directory: str, target_directory: str, src_lang: str, trg_lang: str):
    """Recursively translates the filenames in a directory."""
    
    # Define the model and tokenizer
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{trg_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Get the list of all files in directory tree at given path
    list_of_files = []
    for (dirpath, dirnames, filenames) in os.walk(source_directory):
        list_of_files += [os.path.join(dirpath, file) for file in filenames]

    # Walk through the directory
    for file in tqdm(list_of_files, desc='Translating filenames', unit='file'):
        try:
            # Translate the filename
            translated_name = translate(os.path.splitext(os.path.basename(file))[0], model, tokenizer)

            # Keep the extension of the file
            translated_name_with_ext = f'{translated_name}{os.path.splitext(file)[1]}'

            # Define the old and new file paths
            new_file_path = os.path.join(target_directory, translated_name_with_ext)

            # Ensure that the target directory structure exists
            os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

            # Copy the file with the new name
            shutil.copy2(file, new_file_path)

        except Exception as e:
            print(f"Error translating/copying file {file}: {e}")


def main():
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(description='Translate filenames in a directory.')
    parser.add_argument('src_lang', help='The source language (e.g., "en").')
    parser.add_argument('trg_lang', help='The target language (e.g., "es").')
    parser.add_argument('source_directory', help='The source directory path.')
    parser.add_argument('target_directory', help='The target directory path.')
    args = parser.parse_args()

    # Validate the directories
    if not os.path.isdir(args.source_directory):
        print(f"The source directory {args.source_directory} does not exist.")
        return

    if not os.path.isdir(args.target_directory):
        print(f"The target directory {args.target_directory} does not exist.")
        return

    # Run the script
    translate_filenames(args.source_directory, args.target_directory, args.src_lang, args.trg_lang)


if __name__ == "__main__":
    print('''
    _____     _ _       _   _             
   |_   _|__ (_| |_ __ _| |_(_) ___  _ __  
     | |/ _ \| | __/ _` | __| |/ _ \| '_ \ 
     | | (_) | | || (_| | |_| | (_) | | | |
     |_|\___/|_|\__\__,_|\__|_|\___/|_| |_|
    ''')
    main()
