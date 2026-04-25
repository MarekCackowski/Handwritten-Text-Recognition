import os
import shutil
import re
import xml.etree.ElementTree as elementTree

# --- CONFIG ---
XML_FOLDER = r"C:\OCR\xml"
IMAGES_FOLDER = r"C:\OCR\words_data"
OUTPUT_FOLDER = r"C:\OCR\readable_dataset"


def sanitize_filename(text):
    # Remove illegal characters.
    # NOTE: "." becomes "", so "Dr." -> "Dr"
    clean = re.sub(r'[<>:"/\\|?*.]', '', text)
    return clean


def parse_iam_xml(xml_folder):
    mapping = {}
    print("Parsing XML...")
    for root, _, files in os.walk(xml_folder):
        for f in files:
            if f.endswith(".xml"):
                try:
                    tree = elementTree.parse(os.path.join(root, f))
                    for word_node in tree.getroot().findall(".//word"):
                        w_id = word_node.get('id')
                        w_text = word_node.get('text')
                        if w_id and w_text:
                            mapping[w_id] = w_text
                except:
                    pass
    return mapping


def rename_dataset_clean():
    id_to_text = parse_iam_xml(XML_FOLDER)

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print(f"Renaming files into {OUTPUT_FOLDER}...")

    # Track counts to handle duplicates efficiently
    # Format: { "The": 5, "Cat": 2 }
    word_counts = {}

    count = 0
    for root, _, files in os.walk(IMAGES_FOLDER):
        for f in files:
            if f.endswith(".png"):
                file_id = os.path.splitext(f)[0]

                if file_id in id_to_text:
                    word = id_to_text[file_id]
                    clean_word = sanitize_filename(word)

                    # --- COUNTER LOGIC ---
                    if clean_word not in word_counts:
                        word_counts[clean_word] = 0
                        new_filename = f"{clean_word}.png"
                    else:
                        word_counts[clean_word] += 1
                        # Create "Word_1.png", "Word_2.png"
                        new_filename = f"{clean_word}_{word_counts[clean_word]}.png"
                    # ---------------------

                    src = os.path.join(root, f)
                    dst = os.path.join(OUTPUT_FOLDER, new_filename)

                    shutil.copy(src, dst)
                    count += 1

                    if count % 1000 == 0:
                        print(f"Processed {count} images...")

    print("Done!")

if __name__ == "__main__":
    rename_dataset_clean()
