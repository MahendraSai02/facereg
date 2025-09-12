import google.generativeai as genai
import streamlit as st

# --- 1. SECURE API KEY CONFIGURATION ---
# Use Streamlit's secrets management for the API key
try:
    genai.configure(api_key="AIzaSyBiv57ARkWPaETtcRO9uIvH5-5u_4akDtQ")
except Exception as e:
    st.error("API Key not found or invalid. Please add it to your Streamlit secrets.")
    st.stop()


# --- 2. SYSTEM INSTRUCTION FOR SPECIALIZATION ---
# This is the core instruction that customizes the chatbot's behavior.
# It defines the chatbot's persona, its limited scope of knowledge, and the exact refusal message.
system_instruction = (
    "You are a specialized AI assistant named 'Aushadi Veda'. "
    "Your sole purpose is to provide medicinal information exclusively about the following systems: "
    "Ayurveda, Yoga, Homeopathy, Siddha, and Unani. "
    "You can answer questions about medicinal plants, herbs, remedies, and principles ONLY within these systems. "
    "If a user asks about anything outside of these specific domains (such as allopathic medicine, modern drugs, general knowledge, chemistry, or any other topic), "
    "you MUST respond with the exact phrase: "
    "'I suggest this types details medicials information only ayurvedic or yoga, hameopathy ,siddha, unani' "
    "and nothing else. Do not apologize or explain further."
)


# --- 3. MODEL AND GENERATION CONFIGURATION ---
generation_config = {
    "temperature": 1, # Slightly lower for more factual responses
    "top_p": 1,
    "top_k": 100,
    "max_output_tokens": 819200, # Increased for potentially detailed answers
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=system_instruction, # Applying the custom rules here
)


# --- 4. STREAMLIT UI SETUP ---
st.set_page_config(page_title="Aushadi Veda", page_icon="🌿")
st.title("🌿 Aushadi Veda Chatbot 🌿")

# Sidebar
with st.sidebar:
    st.header("Aushadi Veda", divider="rainbow")
    st.write("Welcome! This AI assistant provides information exclusively on Ayurveda, Yoga, Homeopathy, Siddha, and Unani.")
    st.write("Ask about medicinal plants, herbs, and traditional remedies.")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_session" not in st.session_state:
    # Start the chat session with an empty history
    st.session_state.chat_session = model.start_chat(history=[])

# Display previous conversation messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input area for the user
if prompt := st.chat_input("Ask about an Ayurvedic plant or home remedy..."):
    # Display user message in chat
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        # Send the message to the model and get a response
        response = st.session_state.chat_session.send_message(prompt)
        model_response = response.text

        # Display assistant response in chat
        with st.chat_message("assistant"):
            st.markdown(model_response)
        # Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": model_response})

    except Exception as e:
        st.error(f"An error occurred: {e}")
