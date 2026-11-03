from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

def load_all_documents(data_dir: str) -> List[Any]:
    data_path = Path(data_dir)
    if not data_path.is_absolute() and not data_path.exists():
        data_path = Path(__file__).resolve().parent.parent / data_path
    data_path = data_path.resolve()
    print(f"Data path: {data_path}")
    documents = []

    if not data_path.exists():
        print(f"Data directory does not exist: {data_path}")
        return documents

    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
    for pdf_file in pdf_files:
        print(f"Loading PDF: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            print(loader)
            loaded = loader.load()
            print(f"Loaded {len(loaded)} PDF docs from {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"Failed to load PDF {pdf_file}: {e}")

    # TXT files
    txt_files = list(data_path.glob('**/*.txt'))
    print(f"Found {len(txt_files)} TXT files: {[str(f) for f in txt_files]}")
    for txt_file in txt_files:
        print(f"Loading TXT: {txt_file}")
        try:
            loader = TextLoader(str(txt_file))
            loaded = loader.load()
            print(f"Loaded {len(loaded)} TXT docs from {txt_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"Failed to load TXT {txt_file}: {e}")

    csv_files = list(data_path.glob('**/*.csv'))
    print(f"Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")
    for csv_file in csv_files:
        print(f"Loading CSV: {csv_file}")
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            print(f"Loaded {len(loaded)} CSV docs from {csv_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"Failed to load CSV {csv_file}: {e}")

    xlsx_files = list(data_path.glob('**/*.xlsx'))
    print(f"Found {len(xlsx_files)} Excel files: {[str(f) for f in xlsx_files]}")
    for xlsx_file in xlsx_files:
        print(f"Loading Excel: {xlsx_file}")
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            loaded = loader.load()
            print(f"Loaded {len(loaded)} Excel docs from {xlsx_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"Failed to load Excel {xlsx_file}: {e}")

    docx_files = list(data_path.glob('**/*.docx'))
    print(f"Found {len(docx_files)} Word files: {[str(f) for f in docx_files]}")
    for docx_file in docx_files:
        print(f"Loading Word: {docx_file}")
        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()
            print(f"Loaded {len(loaded)} Word docs from {docx_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"Failed to load Word {docx_file}: {e}")

    json_files = list(data_path.glob('**/*.json'))
    print(f"Found {len(json_files)} JSON files: {[str(f) for f in json_files]}")
    for json_file in json_files:
        print(f"Loading JSON: {json_file}")
        try:
            loader = JSONLoader(str(json_file))
            loaded = loader.load()
            print(f"Loaded {len(loaded)} JSON docs from {json_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"Failed to load JSON {json_file}: {e}")

    print(f"Total loaded documents: {len(documents)}")
    return documents

if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    if docs:
        sample = docs[0]
        page_content = getattr(sample, "page_content", str(sample))
        preview = str(page_content)[:200].encode("ascii", errors="replace").decode("ascii")
        print("Example document preview:", preview)
    else:
        print("Example document: None")
