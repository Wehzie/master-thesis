import algo

class Reinforce(algo.SearchAlgo):
    """
    use information about rejecting a candidate to direct the search

    Actions:
        - draw weight
        - draw oscillator
        - adjust weight distribution
        - adjust oscillator distribution
    Reward:
        - delta rmse
    Utility:
        - rmse
    """
