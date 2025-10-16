import streamlit as st
from chatbot import get_answer

# Page config
st.set_page_config(page_title="Kasturi Assist Chatbot", page_icon="💬")
st.title("💬 Kasturi Assist Chatbot")
st.write("Ask me anything about Kasturi Assist!")

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input box
user_input = st.text_input("Type your question:")

# Buttons: Send / Clear
col1, col2 = st.columns([1,1])
send_clicked = col1.button("Send")
clear_clicked = col2.button("Clear Chat")

# Handle sending message
if send_clicked and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = get_answer(user_input)
    st.session_state.messages.append({"role": "bot", "content": response})

# Handle clearing chat
if clear_clicked:
    st.session_state.messages = []

# Display chat messages after updates
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**🧑 You:** {msg['content']}")
    else:
        st.markdown(f"**🤖 Bot:** {msg['content']}")
