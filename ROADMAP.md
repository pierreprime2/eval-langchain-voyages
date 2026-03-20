# Roadmap - Chatbot Agent de Voyage

## Spécification OpenSpec

### 1. Vue d'ensemble

**Projet** : Chatbot agent de voyage avec LangGraph
**Objectif** : Aider les utilisateurs à choisir des voyages et logements parmi une liste prédéfinie, en fonction de critères extraits de leurs messages.
**Stack** : Python 3.11, LangChain, LangGraph, Gemini (Google AI Studio)

### 2. Architecture

```
[Utilisateur] → message → [LangGraph Agent]
                                │
                    ┌───────────┴───────────┐
                    │                       │
              [Node: extract]         [Node: respond]
              Extraction des          Matching voyages
              critères via           + génération réponse
              structured output
                    │                       │
                    └───────────┬───────────┘
                                │
                          [State]
                          - user_message
                          - ai_message
                          - criteres (plage, montagne,
                            ville, sport, detente,
                            acces_handicap)
```

### 3. Étapes d'implémentation

#### Phase 1 : Setup projet
- [ ] Initialiser le projet Python (pyproject.toml, dépendances)
- [ ] Configurer .env pour les clés API (Gemini)
- [ ] Configurer .gitignore
- [ ] Créer le repo GitHub public

#### Phase 2 : Données et modèles
- [ ] Définir le TypedDict `Criteres` pour la sortie structurée
- [ ] Définir le TypedDict `AgentState` pour le state LangGraph
- [ ] Stocker les données des voyages en JSON

#### Phase 3 : Agent LangGraph
- [ ] Node `extract_criteria` : extraction des critères via structured output LLM
- [ ] Node `respond` : matching des voyages + génération de la réponse
- [ ] Routing conditionnel (critères remplis ou non)
- [ ] Compilation du graph

#### Phase 4 : Configuration & Déploiement
- [ ] Fichier `langgraph.json` pour `langgraph dev`
- [ ] Tests manuels avec exemples de conversations
- [ ] Déploiement sur Render (optionnel)

### 4. Critères utilisateur

| Critère | Type | Défaut |
|---------|------|--------|
| plage | Optional[bool] | None |
| montagne | Optional[bool] | None |
| ville | Optional[bool] | None |
| sport | Optional[bool] | None |
| detente | Optional[bool] | None |
| acces_handicap | Optional[bool] | None |

### 5. Données voyages

| Voyage | Labels | Accessible |
|--------|--------|------------|
| Randonnée camping en Lozère | sport, montagne, campagne | Non |
| 5 étoiles à Chamonix option fondue | montagne, détente | Oui |
| 5 étoiles à Chamonix option ski | montagne, sport | Non |
| Palavas de paillotes en paillotes | plage, ville, détente, paillote | Oui |
| 5 étoiles en rase campagne | campagne, détente | Oui |

### 6. Flux de l'agent

1. L'utilisateur envoie un message
2. **Node extract** : Le LLM analyse le message et extrait les critères (structured output)
3. Les critères sont fusionnés avec les critères existants dans le state
4. **Node respond** :
   - Si aucun critère rempli → demander des précisions
   - Si critères remplis → matcher les voyages et proposer le meilleur match
5. Retour au step 1
