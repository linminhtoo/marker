from typing import TypedDict, List, Dict


class BlockScores(TypedDict):
    score: float
    specific_scores: Dict[str, float | List[float]]
