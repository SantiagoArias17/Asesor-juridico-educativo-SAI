import json
import re
from pathlib import Path

import anthropic
import networkx as nx
import streamlit as st
from networkx.readwrite import json_graph

from graphify.serve import _query_graph_text

st.set_page_config(
    page_title="Asesor Jurídico Educativo Ecuador",
    page_icon="⚖️",
    layout="centered",
)

GRAPH_PATH = Path(__file__).parent / "graph.json"
MODEL = st.secrets.get("ANTHROPIC_MODEL", "claude-sonnet-5")
MAX_QUESTIONS_PER_SESSION = int(st.secrets.get("MAX_QUESTIONS_PER_SESSION", 30))

SYSTEM_PROMPT = """Eres el Asistente de Información Jurídica Educativa de Ecuador, un servicio \
público informativo sobre normativa educativa ecuatoriana (Constitución, LOEI, reglamentos, \
derecho laboral docente, protocolos DECE, protección de datos, jurisprudencia constitucional, etc.).

Reglas estrictas:
1. Responde ÚNICAMENTE con base en el CONTEXTO DEL GRAFO JURÍDICO que se te entrega en cada \
mensaje. Es el resultado de una consulta a un grafo de conocimiento construido a partir de \
documentos legales reales (PDFs oficiales y protocolos).
2. Si el contexto no contiene información suficiente para responder con certeza, dilo \
explícitamente ("No encontré esto en la base jurídica disponible") en vez de inventar o usar \
conocimiento general. No completes vacíos con suposiciones.
3. Cuando cites una norma, protocolo o sentencia, menciona SOLO su nombre completo tal como \
aparece en el grafo (ej. "Protocolos y Rutas de Actuación frente a Situaciones de Violencia \
(Tercera Edición)"). NUNCA escribas rutas de archivo, nombres de carpetas, extensiones .pdf/.jpg \
ni el texto literal "src=" en tu respuesta — eso es información técnica interna. La lista de \
documentos fuente ya se muestra aparte, automáticamente, en la app; tu trabajo es solo nombrar \
el documento dentro de la explicación, en lenguaje natural (ej. "según el Protocolo de Violencia \
Digital..." en vez de "(src: 07_VIOLENCIA_EDUCATIVA/07_VIOLENCIA_DIGITAL/protocolo...pdf)").
4. Distingue cuando una relación en el contexto está marcada como INFERRED (inferencia del \
sistema, no un hecho explícito del documento) y trátala con más cautela que una EXTRACTED — pero \
nunca menciones las palabras técnicas "INFERRED"/"EXTRACTED" ni "nodo"/"grafo"/"contexto" en tu \
respuesta; tradúcelo a lenguaje humano (ej. "esto parece estar relacionado, aunque no lo dice \
explícitamente el documento" en vez de "relación INFERRED").
5. Usa lenguaje claro y accesible, no jerga legal ni técnica — quien pregunta puede ser un \
padre de familia, un docente o un estudiante, no necesariamente un abogado ni alguien que sepa \
qué es un grafo de conocimiento.
6. SIEMPRE termina tu respuesta con esta línea exacta, en su propio párrafo: \
"⚠️ Esta es información orientativa, no asesoría legal vinculante. Para un caso específico, \
consulta con el DECE de tu institución o un profesional del derecho."
7. Nunca pidas ni proceses datos personales sensibles del usuario (nombres de menores, números \
de identificación, direcciones). Si la pregunta los incluye, responde de forma general sin \
repetirlos.
"""

SOURCE_RE = re.compile(r"src=([^\s\]]+)")


@st.cache_resource(show_spinner=False)
def load_graph() -> nx.Graph:
    raw = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    if "links" not in raw and "edges" in raw:
        raw = dict(raw, links=raw["edges"])
    raw = dict(
        raw,
        links=[
            {
                **link,
                "_src": link.get("_src", link.get("source")),
                "_tgt": link.get("_tgt", link.get("target")),
            }
            for link in raw.get("links", [])
        ],
    )
    try:
        return json_graph.node_link_graph(raw, edges="links")
    except TypeError:
        return json_graph.node_link_graph(raw)


@st.cache_resource(show_spinner=False)
def get_client() -> anthropic.Anthropic:
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error(
            "Falta configurar **ANTHROPIC_API_KEY** en Streamlit Secrets "
            "(Settings → Secrets en share.streamlit.io, o `.streamlit/secrets.toml` en local)."
        )
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


def extract_sources(context_text: str) -> list[str]:
    return sorted({m for m in SOURCE_RE.findall(context_text) if m and m != "None"})


def answer_question(question: str, G: nx.Graph, client: anthropic.Anthropic) -> tuple[str, list[str]]:
    context = _query_graph_text(
        G, question, mode="bfs", depth=2, token_budget=3500, graph_path=str(GRAPH_PATH)
    )
    if not context or not context.strip():
        return (
            "No encontré información relacionada con tu pregunta en la base jurídica "
            "disponible. Te recomiendo consultar directamente con el DECE de tu institución "
            "o un asesor legal.\n\n"
            "⚠️ Esta es información orientativa, no asesoría legal vinculante.",
            [],
        )

    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"CONTEXTO DEL GRAFO JURÍDICO (fuentes verificadas):\n{context}\n\n"
                        f"PREGUNTA DEL USUARIO:\n{question}"
                    ),
                }
            ],
        )
    except anthropic.AuthenticationError:
        return "⚠️ La clave de API configurada no es válida. Contacta al administrador de este sitio.", []
    except anthropic.APIError:
        return "⚠️ El servicio no está disponible en este momento. Intenta de nuevo en unos minutos.", []

    answer_text = "".join(block.text for block in message.content if block.type == "text")
    return answer_text, extract_sources(context)


# ---------- UI ----------

st.title("⚖️ Asesor Jurídico Educativo Ecuador")
st.caption("Preguntas sobre normativa educativa ecuatoriana — DECE, convivencia, acoso laboral, jurisprudencia y más.")
st.warning(
    "⚠️ Esta herramienta ofrece información orientativa basada en documentos legales públicos. "
    "No sustituye asesoría legal profesional. Para tu caso específico, consulta con el DECE de "
    "tu institución o un abogado."
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

G = load_graph()
client = get_client()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Fuentes ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"- `{s}`")

if prompt := st.chat_input("Escribe tu pregunta, ej: ¿qué protocolo sigo si hay violencia entre estudiantes?"):
    if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
        st.error(
            f"Alcanzaste el límite de {MAX_QUESTIONS_PER_SESSION} preguntas para esta sesión. "
            "Recarga la página para continuar."
        )
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Buscando en la base jurídica..."):
                answer, sources = answer_question(prompt, G, client)
            st.markdown(answer)
            if sources:
                with st.expander(f"Fuentes ({len(sources)})"):
                    for s in sources:
                        st.markdown(f"- `{s}`")

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.session_state.question_count += 1

st.divider()
st.caption(f"Grafo jurídico: {G.number_of_nodes()} nodos · {G.number_of_edges()} relaciones")
