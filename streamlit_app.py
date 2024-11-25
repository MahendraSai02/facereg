import google.generativeai as genai
import streamlit as st
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# API configuration
genai.configure(api_key="AIzaSyBBwgdrP-e3HyI4bG_KFN1a_c0JzdfiWH4")

# Generation configuration
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 200,
    "response_mime_type": "text/plain",
}

# Create the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,  # type: ignore
)

# Initialize the conversation history
history = []

# Function to interact with the model and return a response
def gemini_1(query):
    chat_session = model.start_chat(
        history=history
    )
    response = chat_session.send_message(query)
    model_response = response.text
    # Update conversation history
    history.append({"role": "user", "parts": [query]})
    history.append({"role": "model", "parts": [model_response]})
    return model_response

# Streamlit UI setup
st.title("Welcome to gemini vanquisher")

# Sidebar for additional information or settings
with st.sidebar:
    st.header("Gemini Vanquisher", divider="rainbow")
    st.write("Welcome to the Gemini Vanquisher AI chat interface!")
    st.write("Ask me anything, about the health related quires ")

# Initialize session state for storing conversation messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input area for the user to send messages
prompt = st.chat_input("How can I assist you today?")

# If the user submits a query, process it and get a response
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get the response from the model
    response = gemini_1(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

