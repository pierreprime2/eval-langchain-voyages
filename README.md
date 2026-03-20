# Agent de Voyage - LangGraph

Chatbot intelligent qui aide les utilisateurs à trouver leur voyage idéal parmi une sélection prédéfinie, en extrayant leurs préférences via un LLM.

## Stack

- **Python 3.11** / **LangChain** / **LangGraph**
- **Gemini 2.0 Flash** via OpenRouter
- Déployé sur **Render**

## Fonctionnement

1. L'utilisateur envoie un message
2. Le LLM extrait les critères (plage, montagne, ville, sport, détente, accessibilité) via **structured output**
3. Les critères sont fusionnés avec l'état existant
4. L'agent propose les voyages correspondants ou demande des précisions

## Lancer en local

```bash
cp .env.example .env
# Renseigner OPENROUTER_API_KEY dans .env
pip install -e .
langgraph dev
```

## Structure

```
agent/
├── data.py      # Liste des voyages
├── state.py     # State LangGraph (critères + messages)
└── graph.py     # Nodes extract/respond + graph
```
