import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

prompt = PromptTemplate(
    input_variables=["contexto", "pergunta"],
    template=PROMPT_TEMPLATE
)

embeddings = GoogleGenerativeAIEmbeddings(model=f"models/{os.getenv('GEMINI_EMBEDDING_MODEL')}")

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True
)

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.2)

chain = prompt | llm

RELEVANCE_THRESHOLD = 0.5

def search_prompt(question=None):
    if not question:
        return "Por favor, insira uma pergunta para consultar a base de conhecimento."
    results = store.similarity_search_with_relevance_scores(question, k=10)
    filtered = [(doc, score) for doc, score in results if score >= RELEVANCE_THRESHOLD]
    if not filtered:
        return "Não foram encontrados documentos relevantes para sua pergunta."
    context = "\n\n".join([f"Documento {i+1} (relevância: {score:.2f}):\n{doc.page_content.strip()}" for i, (doc, score) in enumerate(filtered)])
    return chain.invoke({"contexto": context, "pergunta": question})

def display_menu():
    print("\n=== MENU ===")
    print("1. Consultar base de conhecimento")
    print("2. Sair")

clear_screen = lambda: print("\033[H\033[J", end="")

def display_menu_options():
    while True:
        display_menu()
        choice = input("Escolha uma opção: ")
        if choice == '1':
            question = input("Digite sua pergunta: ")
            print("\nGerando resposta...")
            result = search_prompt(question)
            print(result.content)

        elif choice == '2':
            print("Saindo...")
            break
        else:
            print("Opção inválida. Tente novamente.")
            clear_screen()

if __name__ == "__main__":
    display_menu_options()