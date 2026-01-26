from langchain_text_splitters import CharacterTextSplitter

text = """
      Artificial Intelligence (AI) is transforming the way humans interact with technology. From simple rule-based systems to complex neural networks, AI has evolved rapidly over the past few decades. Today, AI is used in healthcare to assist doctors in diagnosing diseases, in finance to detect fraudulent transactions, and in education to personalize learning experiences for students.

One of the biggest challenges in building AI systems is handling large amounts of text data. Text must often be cleaned, processed, and split into smaller chunks before it can be analyzed by machine learning models or large language models. This process is known as text preprocessing, and it plays a crucial role in ensuring accurate and efficient results.

Character splitting is a basic but important technique in text preprocessing. It involves dividing a long piece of text into smaller segments based on a fixed number of characters. This approach is commonly used when working with APIs or models that have context-length limits. If text is not split properly, important information may be lost or truncated.

However, character splitting has limitations. Splitting text blindly by character count can cut sentences or ideas in half, reducing semantic meaning. To solve this, developers often use overlapping chunks or combine character splitting with sentence or token-based splitting strategies.

Understanding how and when to use character splitting is essential for anyone working with natural language processing, retrieval-augmented generation (RAG), or document-based AI systems.
"""


splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator = ""

)    

result = splitter.split_text(text)

for i , c in enumerate(result):
    print(f"Chunk {i+1} , ({len(c)} chars) : \n{c}")