from langchain_community.vectorstores import Chroma

# Path to the vector database
# Modify this path if the database is located elsewhere
db_path = "./rag_db"

# Load the vector database for inspection
vectordb = Chroma(persist_directory=db_path, embedding_function=None)

# Retrieve all documents and their metadata from the database
all_docs = vectordb._collection.get(include=["metadatas", "documents"])

# Iterate through the documents and metadata to inspect chunks
for i, (content, meta) in enumerate(zip(all_docs["documents"], all_docs["metadatas"])):
    # Check if the chunk text contains "Section 33" at the start or as a header
    if "SECTION 33" in content.upper() or "SEC. 33" in content.upper():
        print(f"\n--- Section 33 Chunk {i+1} ---")
        print("Content:")
        print(content)
        print("\nMetadata:")
        print(meta)

# Save all chunks to a file for further inspection
output_file = "chunks_inspection_output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for i, (content, meta) in enumerate(zip(all_docs["documents"], all_docs["metadatas"])):
        f.write(f"\n--- Chunk {i+1} ---\n")
        f.write("Content:\n")
        f.write(content + "\n")
        f.write("\nMetadata:\n")
        f.write(str(meta) + "\n")

print(f"\nAll chunks have been saved to {output_file} for further inspection.")
