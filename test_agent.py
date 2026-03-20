"""Script de test manuel de l'agent de voyage."""
from dotenv import load_dotenv

load_dotenv()

from agent.graph import graph

def test_message(message: str, state: dict = None):
    """Teste un message utilisateur et affiche le résultat."""
    print(f"\n{'='*60}")
    print(f"USER: {message}")
    print(f"{'='*60}")

    if state is None:
        state = {
            "user_message": message,
            "ai_message": "",
            "criteres": {},
        }
    else:
        state["user_message"] = message

    result = graph.invoke(state)

    print(f"\nCritères extraits: {result['criteres']}")
    print(f"\nAI: {result['ai_message']}")
    print(f"{'='*60}")

    return result


if __name__ == "__main__":
    # Test 1 : Message sans critère
    print("\n>>> TEST 1: Message de salutation")
    state1 = test_message("Bonjour !")

    # Test 2 : Message avec critère plage
    print("\n>>> TEST 2: Recherche plage")
    state2 = test_message("Bonjour, je cherche des vacances à la plage")

    # Test 3 : Message incohérent
    print("\n>>> TEST 3: Message incompréhensible")
    state3 = test_message("srrrrzzzhdj")

    # Test 4 : Mise à jour de critères (simulation conversation)
    print("\n>>> TEST 4: Changement de préférence (montagne au lieu de plage)")
    state4 = test_message("Non en fait je préfère la montagne", state2)

    # Test 5 : Critère accessibilité
    print("\n>>> TEST 5: Besoin d'accessibilité handicap")
    state5 = test_message("Je cherche un voyage détente accessible aux personnes handicapées")
