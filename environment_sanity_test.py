"""
Environment sanity check.
Run this to verify all dependencies are correctly installed.
"""
import sys
import os
from dotenv import load_dotenv
from config.settings import settings
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_groq import ChatGroq
from ragas.metrics import faithfulness
import pypdf

load_dotenv()

def check(name, fn):
    try:
        fn()
        print(f"Pass {name}")
        return True
    except Exception as e:
        print(f"Fail {name}: {e}")
        return False
def test_groq():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    r = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[{"role": "user",
                   "content": "Reply with OK only"}],
        max_tokens=10
    )
    assert r.choices[0].message.content is not None

def test_embeddings():
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    emb = model.encode("test")
    assert len(emb) == 384

def test_chromadb():
    client = chromadb.Client()
    col = client.create_collection("sanity_test")
    assert col is not None
    client.delete_collection("sanity_test")

def test_ragas():
    assert faithfulness is not None

def test_langchain():
    assert ChatGroq is not None

def test_pypdf():
    assert pypdf is not None

def test_config():
    assert settings.GROQ_API_KEY is not None, \
        "GROQ_API_KEY not set in .env"
    assert settings.DEFAULT_LLM_PROVIDER is not None


if __name__ == "__main__":
    print("\nAI Testing Portfolio — Setup Check")
    print("=" * 45)

    results = []
    print("\nDependencies:")
    results.append(check("Groq", test_groq))
    results.append(check("Sentence Transformers",
                         test_embeddings))
    results.append(check("ChromaDB", test_chromadb))
    results.append(check("RAGAS", test_ragas))
    results.append(check("LangChain + Groq", test_langchain))
    results.append(check("PyPDF", test_pypdf))

    print("\nConfiguration:")
    results.append(check("Config/Settings", test_config))


    print("\n" + "=" * 45)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"All {total} checks passed!")
        print("   Ready to build the project.")
    else:
        print(f"{passed}/{total} checks passed.")
        print("   Fix failing checks before proceeding.")
    print("=" * 45 + "\n")