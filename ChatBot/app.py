from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_tavily import TavilySearch

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

###TOOLS
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
wikipedia_tool.name = "wikipedia"
wikipedia_tool.description = (
    "Use this tool when you need general information or a summary "
    "about any topic, person, place, or concept from Wikipedia."
)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
arxiv_tool.name = "arxiv"
arxiv_tool.description = (
    "Use this tool to search Arxiv when asked about research papers, "
    "academic studies, or scientific/ML topics."
)

tavily_tool = TavilySearch(max_results=3, include_answer=True)
tavily_tool.name = "tavily_search"
tavily_tool.description = (
    "Use this tool to search the web directly when you need current events "
    "or the latest information. Just pass the query, don't add extra filters."
)

tools = [wikipedia_tool, arxiv_tool, tavily_tool]
tool_map = {t.name: t for t in tools}


##LLM SETUP
llm = ChatGroq(model_name="openai/gpt-oss-120b")
llm_with_tools = llm.bind_tools(tools)


##PROMPT
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Use tools when needed to give accurate, "
            "up-to-date answers. If a tool returns an error or no useful result, "
            "say so honestly and answer using your own knowledge instead.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)


##MEMORY
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


###STREAMLIT UI
st.title("Langchain Chatbot!")

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

input_text = st.chat_input("How can I help you?")


def run_tool_safely(selected_tool, tool_args):
    """Tool ko run karta hai aur agar error aaye to crash karne ke bajaye
    error message string return karta hai, taaki LLM ko pata chal jaaye."""
    try:
        return selected_tool.invoke(tool_args)
    except Exception as e:
        return (
            f"[Tool '{selected_tool.name}' failed with error: {e}. "
            f"Please answer the user's question using your own knowledge instead, "
            f"and mention that live data wasn't available.]"
        )


if input_text:
    with st.chat_message("user"):
        st.write(input_text)

    with st.spinner("Thinking..."):
        try:
            formatted_messages = prompt.format_messages(
                chat_history=st.session_state.chat_history,
                question=input_text,
            )

            ai_response = llm_with_tools.invoke(formatted_messages)

            if ai_response.tool_calls:
                formatted_messages.append(ai_response)
                for tool_call in ai_response.tool_calls:
                    selected_tool = tool_map.get(tool_call["name"])
                    if selected_tool is None:
                        tool_output = f"[Unknown tool requested: {tool_call['name']}]"
                    else:
                        tool_output = run_tool_safely(selected_tool, tool_call["args"])
                    formatted_messages.append(
                        ToolMessage(
                            content=str(tool_output),
                            tool_call_id=tool_call["id"],
                        )
                    )
                final_response = llm_with_tools.invoke(formatted_messages)
                answer = final_response.content or (
                    "I couldn't generate a proper answer. Please try rephrasing your question."
                )
            else:
                answer = ai_response.content or (
                    "I couldn't generate a proper answer. Please try rephrasing your question."
                )
        except Exception as e:
            answer = f"Something went wrong while processing your request: {e}"

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.chat_history.append(HumanMessage(content=input_text))
    st.session_state.chat_history.append(AIMessage(content=answer))