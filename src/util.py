def add_str2keys(s: str, d: dict) -> dict:
    """append the given string to each key of a dictionary"""
    return {f"{s}_{k}": v for k, v in d.items()}