from langchain_core.tools import tool

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract second number from first number"""
    return a - b

@tool
def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b