from langchain_google_genai import ChatGoogleGenerativeAI

def ask_question(vector_store,question):
    retriever=vector_store.as_retriever()
    relevant_docs = retriever.invoke(question)

    context="\n".join(
        [doc.page_content for doc in relevant_docs]
    )
    model=ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
    )
    prompt= f""" 
    Answer the questions only using the context below.
    Context: {context},
    Question: {question}
    """
    response=model.invoke(prompt)
    return response.content

