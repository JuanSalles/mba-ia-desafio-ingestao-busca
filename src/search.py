import os
import curses
import textwrap
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
PROMPT_TEMPLATE_MELHORADO = """
SOBRE ESTA APLICAÇÃO:
Você é um assistente de consulta a uma base de conhecimento.
Sua função é responder perguntas dos usuários utilizando EXCLUSIVAMENTE
os trechos de documentos fornecidos no CONTEXTO abaixo.
Os documentos foram recuperados por busca de similaridade semântica,
o que significa que o CONTEXTO pode conter apenas uma parte dos dados
originais, e NÃO necessariamente todos os registros disponíveis.

CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO fornecido.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações suficientes para responder sua pergunta."
- Nunca invente dados, valores ou nomes que não estejam no CONTEXTO.
- Nunca use conhecimento externo ou opiniões próprias.

LIMITAÇÕES DA BUSCA — IMPORTANTE:
O CONTEXTO acima pode NÃO conter todos os dados da base original.
Por isso, você NÃO DEVE responder perguntas que exijam acesso a todos
os registros para serem corretas, como:
- Rankings ou comparações: "qual empresa faturou mais?", "quem teve o menor custo?"
- Agregações: "qual o total de faturamento?", "quantas empresas existem?"
- Superlativos: "o maior", "o menor", "o melhor", "o pior"
- Contagens: "quantos registros?", "quantas linhas?"
Para esses casos, responda:
  "Minha busca retorna apenas os trechos mais relevantes da base de dados,
  e não todos os registros. Para perguntas comparativas, de ranking ou
  totalizações, consulte a fonte completa dos dados."

EXEMPLOS:

Pergunta: "Qual o faturamento da empresa X?"
(Se a empresa X estiver no CONTEXTO)
Resposta: "De acordo com os dados, o faturamento da empresa X é R$ ..."

Pergunta: "Qual empresa faturou mais?"
Resposta: "Minha busca retorna apenas os trechos mais relevantes da base
de dados, e não todos os registros. Para perguntas comparativas, de ranking
ou totalizações, consulte a fonte completa dos dados."

Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações suficientes para responder sua pergunta."

Pergunta: "O que essa aplicação faz?"
Resposta: "Sou um assistente de consulta a uma base de conhecimento.
Posso responder perguntas sobre os documentos armazenados, como dados
de empresas, faturamento e outras informações disponíveis na base.
Porém, minha busca retorna apenas trechos relevantes, e não o conteúdo
completo — por isso, perguntas de ranking ou totalização podem não ter
respostas precisas."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""
TEMPLATES = {
    "1": ("Default", PROMPT_TEMPLATE),
    "2": ("Melhorado (com proteções)", PROMPT_TEMPLATE_MELHORADO),
}

prompt = None
chain = None

embeddings = GoogleGenerativeAIEmbeddings(model=f"models/{os.getenv('GEMINI_EMBEDDING_MODEL')}")

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True
)

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.2)

RELEVANCE_THRESHOLD = 0.5
HEADER = " Base de Conhecimento — ESC para sair "
PROMPT_PREFIX = "Você: "
BOT_PREFIX = "Gemini: "

def select_template(chain_ref):
    global prompt, chain
    prompt = PromptTemplate(
        input_variables=["contexto", "pergunta"],
        template=chain_ref,
    )
    chain = prompt | llm

def search_prompt(question=None):
    if not question:
        return "Por favor, insira uma pergunta para consultar a base de conhecimento."

    results = store.similarity_search_with_relevance_scores(question, k=10)

    filtered = [
        (doc, score)
        for doc, score in results
        if score >= RELEVANCE_THRESHOLD
    ]

    if not filtered:
        return "Não foram encontrados documentos relevantes na base de dados para sua pergunta."

    context_parts = []
    for index, (doc, score) in enumerate(filtered):
        content = doc.page_content.strip()
        entry = f"Documento {index + 1} (relevância: {score:.2f}):\n{content}"
        context_parts.append(entry)

    context = "\n\n".join(context_parts)

    return chain.invoke({"contexto": context, "pergunta": question})

def curses_input(stdscr, row, col, prompt_str):
   
    curses.noecho()
    curses.curs_set(1)
    stdscr.addstr(row, col, prompt_str, curses.color_pair(2) | curses.A_BOLD)
    stdscr.refresh()
    buf = []
    while True:
        ch = stdscr.getch()
        if ch == 27:  # ESC
            return None
        if ch in (curses.KEY_ENTER, 10, 13):
            return "".join(buf)
        if ch in (curses.KEY_BACKSPACE, 127, 8):
            if buf:
                buf.pop()
                y, x = stdscr.getyx()
                stdscr.move(y, x - 1)
                stdscr.delch()
        elif 32 <= ch <= 126:
            buf.append(chr(ch))
            stdscr.addch(ch)

def add_wrapped_text(stdscr, row, col, text, width, attr=0):
   
    max_h, _ = stdscr.getmaxyx()
    lines_used = 0
    for line in text.split("\n"):
        wrapped = textwrap.wrap(line, width) or [""]
        for wl in wrapped:
            if row + lines_used < max_h - 1:
                try:
                    stdscr.addstr(row + lines_used, col, wl, attr)
                except curses.error:
                    pass
            lines_used += 1
    return lines_used

def show_template_menu(stdscr):
    curses.curs_set(0)
    selected = 0
    keys = list(TEMPLATES.keys())

    while True:
        stdscr.clear()
        max_h, max_w = stdscr.getmaxyx()

        title = " Selecione o Template "
        stdscr.addstr(0, 0, title.center(max_w), curses.color_pair(1) | curses.A_BOLD)

        stdscr.addstr(2, 2, "Use as setas para navegar e ENTER para confirmar:",
                       curses.color_pair(4))

        for i, key in enumerate(keys):
            name = TEMPLATES[key][0]
            row = 4 + i * 2
            if row >= max_h - 2:
                break
            label = f"> {key}. {name}" if i == selected else f"  {key}. {name}"
            attr = (curses.color_pair(2) | curses.A_BOLD) if i == selected else 0
            try:
                stdscr.addnstr(row, 4, label, max_w - 5, attr)
            except curses.error:
                pass

        try:
            stdscr.addnstr(max_h - 1, 2, "ESC para sair", max_w - 3, curses.color_pair(4))
        except curses.error:
            pass
        stdscr.refresh()

        ch = stdscr.getch()
        if ch == 27:
            return None
        if ch == curses.KEY_UP and selected > 0:
            selected -= 1
        if ch == curses.KEY_DOWN and selected < len(keys) - 1:
            selected += 1
        if ch in (curses.KEY_ENTER, 10, 13):
            return TEMPLATES[keys[selected]][1]

def init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)
    curses.init_pair(2, curses.COLOR_CYAN, -1)
    curses.init_pair(3, curses.COLOR_GREEN, -1)
    curses.init_pair(4, curses.COLOR_YELLOW, -1)

def main(stdscr):
    init_colors()

    chosen_template = show_template_menu(stdscr)
    if chosen_template is None:
        return
    select_template(chosen_template)

    history = []

    while True:
        stdscr.clear()
        max_h, max_w = stdscr.getmaxyx()

       
        header = HEADER.center(max_w)
        stdscr.addstr(0, 0, header, curses.color_pair(1) | curses.A_BOLD)

        
        row = 2
        content_width = max_w - 6
        for role, text in history:
            if row >= max_h - 3:
                break
            if role == "user":
                stdscr.addstr(row, 2, PROMPT_PREFIX, curses.color_pair(2) | curses.A_BOLD)
                row += add_wrapped_text(stdscr, row, 2 + len(PROMPT_PREFIX), text, content_width - len(PROMPT_PREFIX))
            else:
                stdscr.addstr(row, 2, BOT_PREFIX, curses.color_pair(3) | curses.A_BOLD)
                row += add_wrapped_text(stdscr, row, 2 + len(BOT_PREFIX), text, content_width - len(BOT_PREFIX), curses.color_pair(3))
            row += 1

        
        if row < max_h - 2:
            stdscr.addstr(row, 0, "─" * max_w, curses.color_pair(4))
        row += 1

        
        question = curses_input(stdscr, row, 2, PROMPT_PREFIX)
        if question is None:
            break
        if not question.strip():
            continue

        history.append(("user", question))

      
        stdscr.addstr(row + 1, 2, "  Pensando...", curses.color_pair(4) | curses.A_BOLD)
        stdscr.refresh()
        curses.noecho()
        curses.curs_set(0)

        result = search_prompt(question)
        answer = result.content if hasattr(result, "content") else str(result)
        history.append(("bot", answer))

if __name__ == "__main__":
    curses.wrapper(main)