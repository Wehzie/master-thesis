import algo

class Reinforce(algo.SearchAlgo):
    """
    use information about rejecting a candidate to direct the search

    Actions:
        - draw weight
        - draw oscillator
        - set row to 0
        - adjust j_replace (number of rows to manipulate at a time)
    Reward:
        - delta rmse
    Utility:
        - rmse
    """
