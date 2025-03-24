import csv
import re

input_file = "src/data/movies_db.csv"
output_file = "src/data/movies_cleaned.csv"

with open(input_file, mode="r", encoding="utf-8") as infile, open(output_file, mode="w", encoding="utf-8", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Read and write the header
    header = next(reader)
    writer.writerow(header)

    for row in reader:
        # Fix incorrect quotes in MovieName
        row[0] = row[0].strip("'\"")  # Remove leading/trailing quotes if inconsistent
        
        # Remove extra commas in Genre
        row[2] = re.sub(r",+", "|", row[2])  # Replace multiple commas with a single "|"

        # Write cleaned row to the new file
        writer.writerow(row)

print(f"Cleaned CSV saved as {output_file}")