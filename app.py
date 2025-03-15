import streamlit as st
import uuid
import datetime
import re
import logging
from typing import List, Dict, Any
from datetime import datetime

from main import run_academic_agent  # Importe la fonction principale de votre agent amélioré
from langgraph.checkpoint.memory import MemorySaver

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration de la page avec un thème moderne et minimaliste
st.set_page_config(
    page_title="📚 Academic Research Assistant",
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un thème élégant et professionnel
st.markdown("""
<style>
    /* Thème global avec palette professionnelle */
    body {
        color: #333;
        background-color: #f5f5f5;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Cacher le header Streamlit */
    header {
        visibility: block;
    }

    /* Conteneur principal avec espace pour chat input fixé */
    .main .block-container {
        padding-bottom: 80px;
    }

    /* Style des messages utilisateur */
    .st-emotion-cache-jdmyp2, div[data-testid="stChatMessageContent"] > div:nth-child(1) {
        background-color: ##f5f5f5;
        border-radius: 10px;
        border-bottom-right-radius: 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        padding: 15px;
    }

    /* Style des messages assistant */
    .st-emotion-cache-19rxjzo, div[data-testid="stChatMessageContent"] > div:nth-child(2) {
        background-color: #f1f1f1;
        border-radius: 10px;
        border-bottom-left-radius: 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        padding: 15px;
    }

    /* Barre latérale */
    .css-1d391kg, div[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #eaeaea;
    }

    /* Personnalisation des boutons */
    .stButton>button {
        border-radius: 6px;
        border: 1px solid #e0e0e0;
        background-color: white;
        color: #555;
        transition: all 0.2s;
        font-weight: 500;
    }

    .stButton>button:hover {
        background-color: #f5f5f5;
        border-color: #c0c0c0;
    }

    .primary-button>button {
        background-color: #4285f4;
        color: white;
        border: none;
    }

    .primary-button>button:hover {
        background-color: #3367d6;
    }

    /* Conversation cards */
    .conversation-card {
        background-color: white;
        padding: 10px 15px;
        border-radius: 6px;
        margin-bottom: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        cursor: pointer;
        transition: all 0.2s;
        border-left: 3px solid transparent;
    }

    .conversation-card:hover {
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    .conversation-card.active {
        border-left-color: #4285f4;
    }

    .conversation-title {
        font-weight: 500;
        color: #333;
        margin: 0;
    }

    .conversation-time {
        font-size: 12px;
        color: #888;
        margin: 0;
    }

    /* Citations et références */
    blockquote {
        border-left: 3px solid #4285f4;
        padding-left: 1rem;
        color: #555;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 0 6px 6px 0;
    }
    
    /* Styling pour le raisonnement ReAct */
    .reasoning-container {
        background-color: #fff8e1;
        border-radius: 8px;
        padding: 12px 15px;
        margin: 10px 0;
        border: 1px solid #ffe082;
        font-size: 14px;
        color: #7b5800;
        font-family: monospace;
    }
    
    .reasoning-title {
        font-weight: 600;
        margin-bottom: 5px;
        color: #ff9800;
    }

    /* Badge d'état de la recherche */
    .search-badge {
        background-color: #e8f0fe;
        color: #4285f4;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
        margin-left: 10px;
    }

    /* Conteneur de message avec marge */
    .message-container {
        margin-bottom: 15px;
    }

    /* En-tête de l'application */
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }

    .header-title {
        margin: 0;
        font-weight: 600;
    }

    .header-subtitle {
        color: #666;
        font-weight: normal;
    }
    
    /* Styling pour l'indicateur de chargement */
    .loading-dots {
        display: inline-block;
    }
    
    .loading-dots span {
        animation: loading 1.4s infinite both;
        background-color: #666;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        display: inline-block;
        margin: 0 2px;
    }
    
    .loading-dots span:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .loading-dots span:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes loading {
        0% {
            opacity: 0.2;
        }
        20% {
            opacity: 1;
        }
        100% {
            opacity: 0.2;
        }
    }
    
    /* Citations en gras */
    .citation {
        font-weight: bold;
        color: #1a73e8;
    }
    
    /* Container pour l'écran d'accueil */
    .welcome-container {
        text-align: center;
        padding: 30px 0;
        max-width: 600px;
        margin: 0 auto;
    }
    
    .welcome-logo {
        font-size: 64px;
        margin-bottom: 20px;
    }
    
    .welcome-title {
        font-size: 24px;
        margin-bottom: 15px;
        font-weight: 600;
    }
    
    .welcome-description {
        color: #666;
        margin-bottom: 30px;
        line-height: 1.6;
    }
    
    /* Status widget customization */
    div[data-testid="stStatus"] {
        background-color: white !important;
        border: 1px solid #eaeaea !important;
        border-radius: 6px !important;
    }
    
    /* Section références en bas de la réponse */
    .references-section {
        margin-top: 20px;
        padding-top: 10px;
        border-top: 1px solid #e0e0e0;
        font-size: 14px;
    }
    
    .references-title {
        font-weight: 600;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# Fonction pour formater le raisonnement ReAct
def format_reasoning(reasoning_tuple):
    if isinstance(reasoning_tuple, tuple) and len(reasoning_tuple) >= 2:
        action_type, content = reasoning_tuple[0], reasoning_tuple[1]
        if action_type == "Thought":
            return f"🤔 **Thinking:** {content}"
        elif action_type == "Action":
            return f"🔍 **Action:** {content}"
        elif action_type == "Observation":
            return f"📊 **Observation:** {content}"
        else:
            return f"**{action_type}:** {content}"
    return str(reasoning_tuple)


# Initialisation de l'état de session pour plusieurs conversations
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "show_reasoning" not in st.session_state:
    st.session_state.show_reasoning = False


# Fonction pour démarrer une nouvelle conversation
def start_new_conversation():
    conversation_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%d %b, %H:%M")
    st.session_state.conversations[conversation_id] = {
        "messages": [{"role": "assistant",
                      "content": "Hello! I'm Melo, your academic research assistant. How can I help with your academic inquiries today?"}],
        "reasonings": [],
        "timestamp": timestamp,
        "title": f"Research {len(st.session_state.conversations) + 1}"
    }
    st.session_state.current_conversation = conversation_id
    st.session_state.editing_title = False


# Fonction pour changer de conversation
def switch_conversation(conversation_id):
    st.session_state.current_conversation = conversation_id
    st.session_state.editing_title = False


# Fonction pour renommer une conversation
def update_conversation_title(conversation_id, new_title):
    if new_title and conversation_id in st.session_state.conversations:
        st.session_state.conversations[conversation_id]["title"] = new_title
    st.session_state.editing_title = False


# Fonction pour supprimer une conversation
def delete_conversation(conversation_id):
    if conversation_id in st.session_state.conversations:
        del st.session_state.conversations[conversation_id]
        if st.session_state.get("current_conversation") == conversation_id:
            if st.session_state.conversations:
                st.session_state.current_conversation = list(st.session_state.conversations.keys())[0]
            else:
                if "current_conversation" in st.session_state:
                    del st.session_state.current_conversation


# Toggle pour afficher/masquer le raisonnement ReAct
def toggle_reasoning():
    st.session_state.show_reasoning = not st.session_state.show_reasoning


# Mise en valeur des citations en gras dans la réponse
def highlight_citations(text):
    # Recherche des citations au format "Author et al. (YEAR)"
    citation_pattern = r'\*\*(.*?)\*\*'
    highlighted_text = re.sub(citation_pattern, r'<span class="citation">\1</span>', text)
    return highlighted_text


# Titre de l'application avec un style moderne
st.markdown(
    '<div class="header-container">'
    '<h1 class="header-title">📚 Academic Research Assistant</h1>'
    '</div>'
    '<p class="header-subtitle">Find precise academic information with conversational AI</p>',
    unsafe_allow_html=True
)

# Sidebar pour la gestion des conversations
with st.sidebar:
    st.markdown("<h3 style='margin-bottom: 20px;'>Your conversations</h3>", unsafe_allow_html=True)

    # Bouton "Nouvelle conversation" avec style amélioré
    st.markdown('<div class="primary-button">', unsafe_allow_html=True)
    if st.button("+ New convversation", use_container_width=True):
        start_new_conversation()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr style='margin: 15px 0; opacity: 0.2;'>", unsafe_allow_html=True)

    # Afficher les conversations existantes avec une UI améliorée
    if not st.session_state.conversations:
        st.info("No ongoing research sessions. Create a new one to start.")

    for conversation_id, conv_data in st.session_state.conversations.items():
        title = conv_data.get("title", f"Research {conversation_id[:6]}")
        timestamp = conv_data.get("timestamp", "")
        is_active = st.session_state.get("current_conversation") == conversation_id

        # Utilisation de colonnes pour afficher la conversation et les boutons
        col1, col2, col3 = st.columns([4, 1, 1])

        with col1:
            if st.button(title, key=f"conv_{conversation_id}", use_container_width=True):
                switch_conversation(conversation_id)

        with col2:
            if st.button("✏️", key=f"edit_{conversation_id}"):
                st.session_state.editing_title = True
                st.session_state.editing_id = conversation_id

        with col3:
            if st.button("🗑️", key=f"delete_{conversation_id}"):
                delete_conversation(conversation_id)

    # Interface de renommage simplifiée
    if st.session_state.get("editing_title", False):
        editing_id = st.session_state.get("editing_id", "")
        if editing_id in st.session_state.conversations:
            current_title = st.session_state.conversations[editing_id].get("title", "")
            new_title = st.text_input("New title:", value=current_title, key="new_title_input")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save"):
                    update_conversation_title(editing_id, new_title)
            with col2:
                if st.button("Cancel"):
                    st.session_state.editing_title = False

# Réserver un espace pour le chat
main_container = st.container()

# Affichage de la conversation courante
if "current_conversation" in st.session_state and st.session_state.current_conversation in st.session_state.conversations:
    current_conversation = st.session_state.current_conversation
    conversation = st.session_state.conversations[current_conversation]
    messages = conversation["messages"]
    reasonings = conversation.get("reasonings", [])
    title = conversation.get("title", f"Research {current_conversation[:6]}")

    with main_container:
        # Afficher l'historique des messages et raisonnements
        for i, message in enumerate(messages):
            with st.chat_message(message["role"]):
                # Appliquer le formatage des citations pour les messages de l'assistant
                if message["role"] == "assistant":
                    st.markdown(highlight_citations(message["content"]), unsafe_allow_html=True)
                else:
                    st.markdown(message["content"])
                
                # Afficher le raisonnement si c'est un message de l'assistant et qu'on a des raisonnements
                if message["role"] == "assistant" and st.session_state.show_reasoning:
                    reasoning_idx = i // 2 - 1 if i > 1 else 0  # Les messages alternent user/assistant
                    if reasoning_idx < len(reasonings) and reasonings[reasoning_idx]:
                        with st.expander("View AI reasoning process"):
                            for reasoning in reasonings[reasoning_idx]:
                                st.markdown(f'<div class="reasoning-container">{format_reasoning(reasoning)}</div>', 
                                           unsafe_allow_html=True)

    # Zone de saisie de l'utilisateur (toujours en bas)
    if user_input := st.chat_input("Ask your academic research question..."):
        # Vérification que le message n'est pas vide
        if user_input.strip():
            messages.append({"role": "user", "content": user_input})

            with main_container:
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    
                    # Affichage immédiat d'un indicateur de chargement
                    message_placeholder.markdown(
                        """<div style="display: flex; align-items: center;">
                        <span>Researching academic sources</span>
                        <div class="loading-dots" style="margin-left: 8px;">
                            <span></span><span></span><span></span>
                        </div>
                        </div>""", 
                        unsafe_allow_html=True
                    )

                    with st.status("Searching academic databases...") as status:
                        try:
                            # Utiliser la fonction de l'agent académique amélioré
                            result = run_academic_agent(user_input, thread_id=current_conversation)
                            
                            # Extraire la réponse et les raisonnements
                            final_response = result["response"]
                            current_reasoning = result["reasonings"]
                            
                            # Afficher la réponse avec mise en forme des citations
                            message_placeholder.markdown(highlight_citations(final_response), unsafe_allow_html=True)
                            
                            # Ajouter la réponse et le raisonnement à l'historique
                            messages.append({"role": "assistant", "content": final_response})
                            reasonings.append(current_reasoning)
                            
                            status.update(label="Research completed", state="complete")

                        except Exception as e:
                            logger.error(f"Erreur lors du traitement de la requête: {str(e)}")
                            status.update(label="Error", state="error")
                            message_placeholder.error(f"An error occurred while processing your research query: {str(e)}")

else:
    # Page d'accueil améliorée quand aucune conversation n'est active
    st.markdown("""
       <div class="welcome-container">
           <div class="welcome-logo">📚</div>
           <h3 class="welcome-title">Welcome to your Academic Research Assistant</h3>
           <p class="welcome-description">
               I'm Melo, your AI research assistant specialized in academic knowledge. I can help you find scholarly articles, 
               summarize research papers, explain complex academic concepts, and guide you through the latest findings in 
               any field of study.
           </p>
           <p class="welcome-description">
               Simply start a new research session and ask me anything related to academic knowledge!
           </p>
       </div>
       """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="primary-button">', unsafe_allow_html=True)
        if st.button("Start your academic research", use_container_width=True):
            start_new_conversation()
        st.markdown('</div>', unsafe_allow_html=True)


# Ajout d'un crédit en bas de page
st.markdown("""
<div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #f5f5f5; padding: 10px; text-align: center; font-size: 12px; color: #666; border-top: 1px solid #eaeaea;">
Powered by advanced academic search | Data sources: Semantic Scholar, Google Scholar
</div>
""", unsafe_allow_html=True)