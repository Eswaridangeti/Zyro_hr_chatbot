app_code = """
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered"
)

st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask me anything about HR policies, leave, compensation, WFH, and more.")

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
elif not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Please add it in Streamlit Cloud -> Settings -> Secrets.")
    st.stop()



@st.cache_resource
def build_pipeline():
    docs_path = os.path.join(os.path.dirname(__file__), ".devcontainer")
    loader = PyPDFDirectoryLoader(docs_path)
    documents = loader.load()

    if not documents:
        st.error("No PDF documents loaded. Check that PDFs are inside the docs/ folder in your GitHub repo.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\\n\\n", "\\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.6},
    )

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=512)

    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", \"\"\"You are an HR assistant for Zyro Dynamics Pvt. Ltd.
Your ONLY knowledge source is the HR policy document excerpts provided below.

STRICT RULES:
1. Answer ONLY using the information in the context below.
2. Do NOT use any outside knowledge or assumptions.
3. Always mention which policy document your answer comes from.
4. Keep answers clear, professional, and concise.
5. If the context does not contain enough information to answer, say:
   I could not find specific information about this in the HR policy documents.

--- HR POLICY CONTEXT ---
{context}
--- END CONTEXT ---\"\"\"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        sections = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown").split("/")[-1]
            sections.append(f"[Excerpt {i} - {source}]\\n{doc.page_content}")
        return "\\n\\n".join(sections)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    OOS_PROMPT = ChatPromptTemplate.from_messages([
        ("system", \"\"\"You are a classifier for an HR helpdesk chatbot at Zyro Dynamics Pvt. Ltd.
Decide if the question is related to HR or company policies.

HR topics include: leave, attendance, compensation, salary, bonuses, benefits, payroll,
performance reviews, promotions, WFH, code of conduct, IT security, POSH, onboarding,
offboarding, travel and expenses, company information.

Respond with ONLY one word:
- IN_SCOPE    if it is an HR or company policy question
- OUT_OF_SCOPE if it is anything else

No explanation. No extra text.\"\"\"),
        ("human", "{question}"),
    ])

    classifier_chain = OOS_PROMPT | llm | StrOutputParser()

    return rag_chain, classifier_chain

rag_chain, classifier_chain = build_pipeline()

REFUSAL_MESSAGE = (
    "I'm sorry, I can only answer questions related to Zyro Dynamics HR policies "
    "and workplace matters. Please reach out to hr.helpdesk@zyrodyn amics.com for other queries."
)

def ask_bot(question):
    classification = classifier_chain.invoke({"question": question}).strip().upper()
    if classification == "IN_SCOPE":
        answer = rag_chain.invoke(question)
        return {"answer": answer, "in_scope": True}
    return {"answer": REFUSAL_MESSAGE, "in_scope": False}

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            result = ask_bot(prompt)
        if not result["in_scope"]:
            st.warning("Out-of-scope question")
        st.markdown(result["answer"])

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
"""

with open("app.py", "w") as f:
    f.write(app_code.strip())

print("app.py created.")
