from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

def load_data(file_path):
    """Load data from a file and return a set of unique (English, Hindi) word pairs."""
    word_pairs = set()
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")  # Assuming tab-separated values
            if len(parts) == 2:
                english, hindi = parts
                # Transliterate Hindi to Romanized Hindi and normalize both words to lowercase
                romanized_hindi = transliterate(hindi, sanscript.DEVANAGARI, sanscript.ITRANS).lower()
                word_pairs.add((english.lower(), romanized_hindi))  # Store as a set to remove duplicates
    return word_pairs

# Load data from both files
file1_data = load_data("dataset/Panlex/hi2.txt")
file2_data = load_data("dataset/Panlex/crowd_transliterations.hi-en.txt")


# Combine both lists
combined_data = file1_data.union(file2_data)

# Write each pair on a new line
with open("en-hi-romanized-dict.txt", "w", encoding="utf-8") as out_file:
    for english, hindi in sorted(combined_data):
        out_file.write(f"{english}\t{hindi}\n")

print("Mered file saved")
