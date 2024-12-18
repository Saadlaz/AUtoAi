from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Custom template filter to get an item from a dictionary."""
    return dictionary.get(key, None)

@register.filter
def multiply(value, factor):
    """Multiply a value by a factor."""
    try:
        return float(value) * float(factor)
    except (ValueError, TypeError):
        return 0
