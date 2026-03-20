from typing import TypedDict, Optional


class Criteres(TypedDict):
    plage: Optional[bool]
    montagne: Optional[bool]
    ville: Optional[bool]
    sport: Optional[bool]
    detente: Optional[bool]
    acces_handicap: Optional[bool]


class AgentState(TypedDict):
    user_message: str
    ai_message: str
    criteres: Criteres
