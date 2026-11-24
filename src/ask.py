from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
llm = ChatAnthropic(model="claude-haiku-5-5")

def ask(question: str, k: int = 4):
    hits = db.similarity_search(question, k=k)
    context = "\n\n".join(
        f"[{h.metadata.get('source', '')}, Seite {h.metadata.get('page', 0) + 1}]\n{h.page_content}"
        for h in hits
    )
    prompt = (
        "Beantworte die Frage nur anhand des Kontexts. "
        "Wenn die Antwort nicht im Kontext steht, sage das ehrlich. "
        "Nenne die Quelle (Dokument, Seite). "
        "Antworte in der Sprache der Frage. Erwähne keine technischen Details über den Kontext (z. B. abgeschnittene Textstellen). Antworte kurz und klar.\n\n"
        f"Kontext:\n{context}\n\nFrage: {question}"
    )
    content = llm.invoke(prompt).content
    if isinstance(content, list):
        answer = "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    else:
        answer = content
    return answer, hits

if __name__ == "__main__":
    answer, _ = ask("Wie oft kann man eine nicht bestandene Prüfung wiederholen?")
    print(answer)
