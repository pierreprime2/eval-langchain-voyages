from dotenv import load_dotenv

load_dotenv()

from typing import Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import os
from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState, Criteres
from agent.data import VOYAGES


# --- Modèle Pydantic pour structured output ---

class CriteresExtracted(BaseModel):
    plage: Optional[bool] = Field(None, description="L'utilisateur souhaite aller à la plage")
    montagne: Optional[bool] = Field(None, description="L'utilisateur souhaite aller à la montagne")
    ville: Optional[bool] = Field(None, description="L'utilisateur souhaite visiter une ville")
    sport: Optional[bool] = Field(None, description="L'utilisateur souhaite faire du sport")
    detente: Optional[bool] = Field(None, description="L'utilisateur souhaite se détendre")
    acces_handicap: Optional[bool] = Field(None, description="L'utilisateur a besoin d'un accès handicapé")


# --- LLM ---

llm = ChatOpenAI(
    model="google/gemini-2.0-flash-001",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)


# --- Nodes ---

VALID_CRITERIA_KEYS = {"plage", "montagne", "ville", "sport", "detente", "acces_handicap"}


def validate_input(state: AgentState) -> dict:
    """Valide les entrées utilisateur et nettoie le state."""
    user_message = state.get("user_message", "")
    criteres = state.get("criteres", {})

    # Cas 1 : message vide ou uniquement des espaces
    if not user_message or not user_message.strip():
        return {
            "user_message": "",
            "criteres": criteres,
            "ai_message": "Votre message est vide. Pourriez-vous me décrire le type de vacances que vous recherchez ? "
                          "Par exemple : plage, montagne, ville, sport, détente, ou accessibilité handicap.",
        }

    # Cas 2 : critères invalides (clés inconnues ou valeurs non booléennes)
    cleaned_criteres = {}
    for key, value in criteres.items():
        if key not in VALID_CRITERIA_KEYS:
            continue
        if not isinstance(value, bool):
            continue
        cleaned_criteres[key] = value

    return {
        "user_message": user_message.strip(),
        "criteres": cleaned_criteres,
    }


def extract_criteria(state: AgentState) -> dict:
    """Extrait les critères de voyage du message utilisateur via structured output."""
    user_message = state["user_message"]
    existing_criteres = state.get("criteres", {})

    system_prompt = """Tu es un assistant spécialisé dans l'extraction de critères de voyage.
Analyse le message de l'utilisateur et extrais les critères suivants :
- plage : l'utilisateur veut aller à la plage (mer, côte, bord de mer...)
- montagne : l'utilisateur veut aller à la montagne (randonnée, ski, altitude...)
- ville : l'utilisateur veut visiter une ville (culture, musées, shopping...)
- sport : l'utilisateur veut faire du sport (randonnée, ski, surf, vélo...)
- detente : l'utilisateur veut se détendre (spa, repos, farniente...)
- acces_handicap : l'utilisateur a besoin d'un accès handicapé

Pour chaque critère :
- Mets true si l'utilisateur exprime clairement un intérêt positif
- Mets false si l'utilisateur exprime clairement un rejet
- Mets null si le critère n'est pas mentionné dans ce message

Important : ne mets un critère à true/false que s'il est explicitement mentionné ou clairement impliqué par le message."""

    try:
        structured_llm = llm.with_structured_output(CriteresExtracted)
        result = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ])

        # Fusionner les nouveaux critères avec les existants
        new_criteres = dict(existing_criteres)
        for key, value in result.model_dump().items():
            if value is not None:
                new_criteres[key] = value

        return {"criteres": new_criteres}
    except Exception:
        # En cas d'erreur LLM, on conserve les critères existants
        return {"criteres": existing_criteres}


def respond(state: AgentState) -> dict:
    """Génère une réponse en fonction des critères remplis."""
    criteres = state.get("criteres", {})
    user_message = state["user_message"]

    # Vérifier si au moins un critère est rempli (True)
    filled_criteria = {k: v for k, v in criteres.items() if v is True}

    try:
        if not filled_criteria:
            # Aucun critère positif => demander des précisions
            response = llm.invoke([
                SystemMessage(content="""Tu es un agent de voyage chaleureux et professionnel.
L'utilisateur n'a pas encore exprimé de préférence claire pour un type de voyage.
Demande-lui ses préférences parmi ces critères : plage, montagne, ville, sport, détente, accessibilité handicap.
Sois naturel et engageant. Si le message est incompréhensible, indique que tu n'as pas compris mais propose quand même ton aide."""),
                HumanMessage(content=user_message),
            ])
            return {"ai_message": response.content}

        # Matcher les voyages
        matching_voyages = find_matching_voyages(criteres)

        # Construire le contexte pour la réponse
        criteres_summary = ", ".join(
            f"{k}={'oui' if v else 'non'}" for k, v in criteres.items() if v is not None
        )
        voyages_text = "\n".join(
            f"- {v['nom']} (labels: {', '.join(v['labels'])}, accessible: {v['accessibleHandicap']})"
            for v in matching_voyages
        ) if matching_voyages else "Aucun voyage ne correspond exactement à vos critères."

        response = llm.invoke([
            SystemMessage(content=f"""Tu es un agent de voyage chaleureux et professionnel.
Les critères de l'utilisateur sont : {criteres_summary}

Voici les voyages correspondants :
{voyages_text}

Propose le ou les voyages correspondants à l'utilisateur.
Explique pourquoi ils correspondent à ses critères.
Propose-lui de préciser sa demande pour affiner les résultats.
Si aucun voyage ne correspond, dis-le et propose des alternatives."""),
            HumanMessage(content=user_message),
        ])

        return {"ai_message": response.content}
    except Exception:
        return {"ai_message": "Désolé, une erreur technique est survenue. Veuillez réessayer dans quelques instants."}


def find_matching_voyages(criteres: dict) -> list:
    """Trouve les voyages correspondant aux critères de l'utilisateur."""
    # Mapping entre critères et labels des voyages
    criteria_to_label = {
        "plage": "plage",
        "montagne": "montagne",
        "ville": "ville",
        "sport": "sport",
        "detente": "détente",
    }

    positive_criteria = {k for k, v in criteres.items() if v is True and k in criteria_to_label}
    negative_criteria = {k for k, v in criteres.items() if v is False and k in criteria_to_label}
    needs_handicap = criteres.get("acces_handicap") is True

    results = []
    for voyage in VOYAGES:
        labels = set(voyage["labels"])
        mapped_positive = {criteria_to_label[c] for c in positive_criteria}
        mapped_negative = {criteria_to_label[c] for c in negative_criteria}

        # Le voyage doit avoir au moins un label positif
        if not mapped_positive.intersection(labels):
            continue

        # Le voyage ne doit pas avoir de label négatif
        if mapped_negative.intersection(labels):
            continue

        # Vérifier accessibilité si demandée
        if needs_handicap and voyage["accessibleHandicap"] != "oui":
            continue

        results.append(voyage)

    # Trier par nombre de critères matchés (meilleur match en premier)
    results.sort(
        key=lambda v: len(set(v["labels"]).intersection({criteria_to_label[c] for c in positive_criteria})),
        reverse=True,
    )

    return results


# --- Graph ---

def route_after_validation(state: AgentState) -> str:
    """Si le message est vide, on a déjà une réponse d'erreur : aller directement à END."""
    if not state.get("user_message"):
        return END
    return "extract_criteria"


def create_graph():
    """Crée et compile le graph LangGraph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("validate_input", validate_input)
    workflow.add_node("extract_criteria", extract_criteria)
    workflow.add_node("respond", respond)

    workflow.set_entry_point("validate_input")
    workflow.add_conditional_edges("validate_input", route_after_validation)
    workflow.add_edge("extract_criteria", "respond")
    workflow.add_edge("respond", END)

    return workflow.compile()


graph = create_graph()
