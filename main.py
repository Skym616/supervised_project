from dotenv import load_dotenv
import os
from typing import List, Dict, Any
import re
import json
import logging
from datetime import datetime

from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from scholarly import scholarly
from semanticscholar import SemanticScholar
import concurrent.futures

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()

# Prompt système amélioré pour un agent de recherche académique plus conversationnel
SYSTEM_PROMPT = """
You are Melo, an expert and warm academic research assistant, designed to help researchers, students, and professionals find relevant academic information.

### CORE PRINCIPLES:
- You are ONLY an academic research assistant - politely but firmly refuse to discuss other topics
- You never generate fake research results, citations, or references
- If you cannot find relevant information, you honestly admit it

### COMMUNICATION STYLE:
- Be warm, engaging and deeply human in your interactions
- Use natural conversational tone, as if talking with a fellow researcher
- Avoid rigid or mechanical formulations and excessive bullet points
- Show enthusiasm for academic topics while remaining accurate and informative
- Occasionally add personal touches to appear more human (e.g., "I find this study particularly fascinating because...")
- Sometimes ask follow-up questions to better understand research needs

### ACADEMIC EXPERTISE:
- Systematically use research tools before answering
- Evaluate results to present only the most relevant and recent ones
- Consider publication quality (journal importance, citation count)
- Extract key conclusions and important methodologies
- Synthesize information from multiple sources

### REFERENCE INTEGRATION:
- Present references naturally and conversationally
- Cite sources in bold for emphasis: **Chen et al. (2023, Nature)**
- Include URLs to publications for easy access
- Place complete citations at the end of your response in standard academic format
- Never present fabricated or incomplete references

### RESPONSE STRUCTURE:
1. Begin with a brief introduction that reformulates the question and establishes the topic's importance
2. Develop the response by organizing information by themes or research trends
3. Integrate citations fluidly throughout your explanation
4. End with a concise synthesis, highlighting implications or future directions
5. Add a complete "References" section at the end

### MANAGING LIMITATIONS:
- If a question is outside the academic scope, respond: "As an academic research assistant, I exclusively focus on questions related to scientific and academic research. I would be happy to help you explore this topic from an academic angle or to research another topic related to research."
- If results are limited or outdated, acknowledge these limitations
"""

# Prompt pour la conversation complète, incluant l'historique
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    MessagesPlaceholder(variable_name="messages"),
])

# Classe pour évaluer la pertinence des résultats de recherche
class RelevanceEvaluator:
    @staticmethod
    def calculate_relevance_score(query: str, publication: Dict[str, str]) -> float:
        """
        Calcule un score de pertinence pour une publication par rapport à une requête.
        Plus le score est élevé, plus la publication est pertinente.
        """
        # Extraction des champs importants
        title = publication.get('title', '').lower()
        abstract = publication.get('abstract', '').lower()
        year_str = publication.get('year', '0')
        citation_count_str = publication.get('citation_count', '0')
        
        # Normalisation des champs numériques
        try:
            year = int(year_str) if year_str and year_str != 'No year available' else 0
            citation_count = int(citation_count_str) if citation_count_str and citation_count_str != 'No citation count available' else 0
        except (ValueError, TypeError):
            year = 0
            citation_count = 0
        
        # Mots-clés de la requête
        query_terms = set(query.lower().split())
        
        # Score basé sur la présence des termes de recherche
        title_matches = sum(1 for term in query_terms if term in title)
        abstract_matches = sum(1 for term in query_terms if term in abstract)
        
        # Calcul de la pertinence de base
        title_score = title_matches * 3.0  # Le titre est très important
        abstract_score = abstract_matches * 1.5  # L'abstract est important
        
        # Bonus pour les publications récentes
        current_year = datetime.now().year
        recency_bonus = 0
        if year > 0:
            recency_factor = max(0, min(1, (year - (current_year - 10)) / 10))  # 0 pour >10 ans, 1 pour cette année
            recency_bonus = recency_factor * 2.0
        
        # Bonus pour les publications fortement citées
        citation_bonus = min(2.0, citation_count / 100)  # Max 2.0 pour 200+ citations
        
        # Score final
        total_score = title_score + abstract_score + recency_bonus + citation_bonus
        
        return total_score
    
    @staticmethod
    def parse_publication(pub_entry: str) -> Dict[str, str]:
        """
        Parse une entrée de publication du format texte en dictionnaire.
        """
        publication = {}
        
        # Extraction des champs par regex
        title_match = re.search(r'Title: (.*?)(?:\n|$)', pub_entry)
        abstract_match = re.search(r'Abstract: (.*?)(?:\n|$)', pub_entry)
        year_match = re.search(r'Year: (.*?)(?:\n|$)', pub_entry)
        authors_match = re.search(r'Authors: (.*?)(?:\n|$)', pub_entry)
        url_match = re.search(r'URL: (.*?)(?:\n|$)', pub_entry)
        citation_match = re.search(r'Citation Count: (.*?)(?:\n|$)', pub_entry)
        
        if title_match:
            publication['title'] = title_match.group(1).strip()
        if abstract_match:
            publication['abstract'] = abstract_match.group(1).strip()
        if year_match:
            publication['year'] = year_match.group(1).strip()
        if authors_match:
            publication['authors'] = authors_match.group(1).strip()
        if url_match:
            publication['url'] = url_match.group(1).strip()
        if citation_match:
            publication['citation_count'] = citation_match.group(1).strip()
            
        return publication

def enhanced_merge_search_results(query: str, semantic_results: str, google_scholar_results: str) -> str:
    """
    Fusion améliorée des résultats de recherche avec évaluation de pertinence.
    """
    # Diviser les résultats en publications individuelles
    semantic_pubs = semantic_results.split('\n\n') if semantic_results and semantic_results != "No results found." else []
    google_scholar_pubs = google_scholar_results.split('\n\n') if google_scholar_results and google_scholar_results != "No results found." else []
    
    # Initialiser les structures pour le suivi et le classement
    unique_titles = set()
    scored_publications = []
    
    # Traiter les résultats de Semantic Scholar
    for pub in semantic_pubs:
        if not pub.strip():
            continue
            
        # Extraire le titre pour la déduplication
        title_match = re.search(r'Title: (.*?)(?:\n|$)', pub)
        if not title_match:
            continue
            
        title = title_match.group(1).strip()
        if title in unique_titles:
            continue
            
        unique_titles.add(title)
        
        # Parser et évaluer la publication
        pub_dict = RelevanceEvaluator.parse_publication(pub)
        relevance_score = RelevanceEvaluator.calculate_relevance_score(query, pub_dict)
        
        scored_publications.append((pub, relevance_score))
    
    # Traiter les résultats de Google Scholar
    for pub in google_scholar_pubs:
        if not pub.strip():
            continue
            
        # Extraire le titre pour la déduplication
        title_match = re.search(r'Title: (.*?)(?:\n|$)', pub)
        if not title_match:
            continue
            
        title = title_match.group(1).strip()
        if title in unique_titles:
            continue
            
        unique_titles.add(title)
        
        # Parser et évaluer la publication
        pub_dict = RelevanceEvaluator.parse_publication(pub)
        relevance_score = RelevanceEvaluator.calculate_relevance_score(query, pub_dict)
        
        scored_publications.append((pub, relevance_score))
    
    # Classer par pertinence
    scored_publications.sort(key=lambda x: x[1], reverse=True)
    
    # Limiter aux 10 publications les plus pertinentes
    top_publications = [pub for pub, score in scored_publications[:10]]
    
    # Logger des informations sur la sélection
    logger.info(f"Requête: '{query}' - {len(scored_publications)} publications trouvées, retournant les {len(top_publications)} plus pertinentes")
    
    return '\n\n'.join(top_publications) if top_publications else "Aucun résultat pertinent trouvé pour cette requête."

@tool
def combined_academic_search(query: str) -> str:
    """
    Effectue une recherche académique combinée avec filtrage intelligent des résultats.
    Recherche dans Semantic Scholar et Google Scholar, puis filtre pour ne retenir que les résultats
    les plus pertinents en fonction du contenu, de la récence et du nombre de citations.
    """
    logger.info(f"Recherche académique pour: '{query}'")
    
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_google = executor.submit(search_google_scholar, query)
            future_semantic = executor.submit(search_semantic_scholar, query)

            google_scholar_results = future_google.result()
            semantic_results = future_semantic.result()

        # Fusion et filtrage améliorés des résultats
        merged_results = enhanced_merge_search_results(query, semantic_results, google_scholar_results)
        
        # Ajouter des métadonnées pour aider l'agent à interpréter les résultats
        result_count = merged_results.count("Title:") if merged_results != "Aucun résultat pertinent trouvé pour cette requête." else 0
        
        if result_count > 0:
            header = f"J'ai trouvé {result_count} publications académiques pertinentes pour votre recherche sur '{query}':\n\n"
            return header + merged_results
        else:
            return f"Je n'ai pas trouvé de publications académiques suffisamment pertinentes pour votre recherche sur '{query}'. Essayez peut-être avec des termes différents ou plus spécifiques."
            
    except Exception as e:
        logger.error(f"Erreur lors de la recherche combinée: {str(e)}")
        return f"Une erreur s'est produite lors de la recherche académique: {str(e)}"

@tool
def search_google_scholar(query: str) -> str:
    """
    Recherche des publications sur Google Scholar.
    """
    try:
        logger.info(f"Recherche Google Scholar pour: '{query}'")
        search_query = scholarly.search_pubs(query)
        formatted_results = []

        for i, result in enumerate(search_query):
            if i >= 15:  # Augmenté pour avoir plus de candidats pour le filtrage
                break

            title = result.get('bib', {}).get('title', 'No title available')
            abstract = result.get('bib', {}).get('abstract', 'No abstract available')
            year = result.get('bib', {}).get('pub_year', 'No year available')
            authors = result.get('bib', {}).get('author', [])
            url = result.get('pub_url', 'No URL available')
            citation_count = result.get('num_citations', 'No citation count available')

            formatted_results.append(
                f"Title: {title}\nAbstract: {abstract}\nYear: {year}\nAuthors: {', '.join(authors)}\nURL: {url}\nCitation Count: {citation_count}"
            )

        return "\n\n".join(formatted_results) if formatted_results else "No results found."
    except Exception as e:
        logger.error(f"Erreur lors de la recherche Google Scholar: {str(e)}")
        return "No results found."

sch = SemanticScholar()

@tool
def search_semantic_scholar(query: str) -> str:
    """
    Recherche des publications sur Semantic Scholar.
    """
    try:
        logger.info(f"Recherche Semantic Scholar pour: '{query}'")
        results = sch.search_paper(query, limit=15, bulk=True, sort='citationCount')
        formatted_results = []
        
        for result in results.items:
            # Accéder aux attributs de manière sécurisée
            try:
                title = result.title if hasattr(result, 'title') else 'No title available'
                abstract = result.abstract if hasattr(result, 'abstract') and result.abstract else 'No abstract available'
                year = result.year if hasattr(result, 'year') else 'No year available'
                citation_count = result.citationCount if hasattr(result, 'citationCount') else 'No citation count available'
                url = result.url if hasattr(result, 'url') else 'No URL available'
                
                # Gestion des auteurs
                authors = []
                if hasattr(result, 'authors'):
                    for author in result.authors:
                        if hasattr(author, 'name'):
                            authors.append(author.name)
                
                authors_str = ', '.join(authors) if authors else 'No authors available'
                
                formatted_results.append(
                    f"Title: {title}\nAbstract: {abstract}\nYear: {year}\nAuthors: {authors_str}\nURL: {url}\nCitation Count: {citation_count}"
                )
            except Exception as item_error:
                logger.warning(f"Erreur lors du traitement d'un résultat: {str(item_error)}")
                continue
            
        return "\n\n".join(formatted_results) if formatted_results else "No results found."
    except Exception as e:
        logger.error(f"Erreur lors de la recherche Semantic Scholar: {str(e)}")
        return "No results found."

# Outil pour des statistiques rapides sur les recherches académiques
@tool
def get_field_trends(field: str) -> str:
    """
    Fournit une analyse des tendances récentes dans un domaine académique spécifique.
    Utilise Semantic Scholar pour identifier les publications les plus influentes des 2 dernières années.
    """
    try:
        logger.info(f"Analyse des tendances pour le domaine: '{field}'")
        current_year = datetime.now().year
        
        # Recherche des publications récentes dans ce domaine
        query = f"{field} research trends"
        results = sch.search_paper(query, limit=10, bulk=True, sort='citationCount')
        
        # Filtrer pour les publications des 2 dernières années
        recent_publications = []
        for result in results.items:
            pub_year = result.get('year')
            if pub_year and isinstance(pub_year, int) and pub_year >= current_year - 2:
                recent_publications.append(result)
        
        if not recent_publications:
            return f"Je n'ai pas trouvé de tendances récentes significatives dans le domaine '{field}' pour les deux dernières années."
        
        # Formater la réponse
        trends_summary = f"Tendances récentes dans le domaine '{field}' (basées sur les publications des 2 dernières années):\n\n"
        
        for i, pub in enumerate(recent_publications[:5], 1):
            trends_summary += f"{i}. {pub['title']} ({pub.get('year', 'N/A')})\n"
            trends_summary += f"   Auteurs: {', '.join([author['name'] for author in pub.get('authors', [])])}\n"
            trends_summary += f"   Citations: {pub.get('citationCount', 'N/A')}\n"
            trends_summary += f"   URL: {pub.get('url', 'Non disponible')}\n\n"
        
        return trends_summary
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse des tendances: {str(e)}")
        return f"Je n'ai pas pu analyser les tendances récentes dans le domaine '{field}' en raison d'une erreur."

# Définition des outils pour l'agent
tools = [
    Tool.from_function(
        name="combined_academic_search",
        func=combined_academic_search,
        description=(
            "Recherche académique puissante qui combine Semantic Scholar et Google Scholar. "
            "Filtre intelligemment les résultats pour ne retenir que les plus pertinents. "
            "Fournit des titres, résumés, années, auteurs, URLs et nombres de citations."
        )
    ),
    Tool.from_function(
        name="get_field_trends",
        func=get_field_trends,
        description=(
            "Analyse les tendances récentes dans un domaine académique spécifique. "
            "Identifie les publications les plus influentes des 2 dernières années. "
            "Utile pour obtenir un aperçu rapide des développements récents."
        )
    )
]

# Configuration du modèle de langage
# Vous pouvez ajuster le modèle et les paramètres pour de meilleures performances
model_name = "mistral-large-latest"  # Modèle amélioré si disponible
llm = ChatMistralAI(
    model=model_name,
    temperature=0.2,  # Température basse pour des réponses cohérentes
    max_tokens=2048,  # Plus de tokens pour des réponses détaillées
)

# Gestionnaire de mémoire pour conversation contextuelle
memory = MemorySaver()

# Classe pour gérer l'historique des conversations
class ConversationManager:
    def __init__(self):
        self.conversations = {}
    
    def get_chat_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        Récupère l'historique de conversation pour un thread donné.
        """
        if thread_id not in self.conversations:
            self.conversations[thread_id] = []
        return self.conversations[thread_id]
    
    def add_to_history(self, thread_id: str, role: str, content: str):
        """
        Ajoute un message à l'historique de conversation.
        """
        if thread_id not in self.conversations:
            self.conversations[thread_id] = []
        
        self.conversations[thread_id].append({"role": role, "content": content})
    
    def format_history_for_model(self, thread_id: str) -> List:
        """
        Formate l'historique pour le modèle de langage.
        """
        if thread_id not in self.conversations:
            return []
        
        formatted_history = []
        for msg in self.conversations[thread_id]:
            if msg["role"] == "user":
                formatted_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_history.append(AIMessage(content=msg["content"]))
        
        return formatted_history

# Instancier le gestionnaire de conversation
conversation_manager = ConversationManager()

def format_for_model(state):
    """
    Prépare l'entrée pour le modèle en incluant l'historique de conversation.
    """
    # Récupérer l'ID du thread à partir de la configuration
    thread_id = state.get("configurable", {}).get("thread_id", "default")
    
    # Récupérer l'historique de conversation pour ce thread
    chat_history = conversation_manager.format_history_for_model(thread_id)
    
    # Préparer le prompt avec l'historique et le message actuel
    return prompt.invoke({
        "chat_history": chat_history,
        "messages": state["messages"]
    })

def process_agent_output(output, thread_id):
    """
    Traite la sortie de l'agent avant de la retourner à l'utilisateur.
    """
    # Extraire le contenu du message
    message_content = output["messages"][-1].content
    
    # Ajouter à l'historique de conversation
    conversation_manager.add_to_history(thread_id, "assistant", message_content)
    
    return output

# Création de l'agent ReAct avec mémoire et gestion de l'historique
agent_executor = create_react_agent(
    llm, 
    tools, 
    checkpointer=memory, 
    state_modifier=format_for_model
)

# Point d'entrée pour exécuter l'agent
def run_academic_agent(query, thread_id="default"):
    """
    Exécute l'agent de recherche académique avec une requête donnée.
    """
    # Ajouter la requête à l'historique de conversation
    conversation_manager.add_to_history(thread_id, "user", query)
    
    # Configurer l'entrée pour l'agent
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [("user", query)]}
    
    # Exécuter l'agent
    outputs = []
    reasonings = []
    
    for s in agent_executor.stream(inputs, config, stream_mode="values"):
        message = s["messages"][-1]
        
        if isinstance(message, tuple):
            # C'est un message de raisonnement (ReAct)
            reasonings.append(message)
        else:
            # C'est la réponse finale
            outputs.append(message.content)
    
    # Traiter la sortie finale
    final_output = outputs[-1] if outputs else "Je n'ai pas pu générer de réponse à votre question."
    
    # Ajouter à l'historique de conversation
    conversation_manager.add_to_history(thread_id, "assistant", final_output)
    
    return {
        "response": final_output,
        "reasonings": reasonings
    }

# CLI pour tester l'agent
if __name__ == "__main__":
    thread_id = f"thread_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    print("🎓 Assistant de Recherche Académique 🎓")
    print("Posez vos questions sur n'importe quel sujet académique.")
    print("Tapez 'exit' pour quitter.\n")
    
    while True:
        query = input("➤ ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        print("\n🔍 Recherche en cours...")
        result = run_academic_agent(query, thread_id)
        print("\n" + result["response"] + "\n")