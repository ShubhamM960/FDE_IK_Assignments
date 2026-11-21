"""
SnackStack Menu Catalog
=======================

The source of truth for every dish SnackStack sells.

How it's used later:
    - tools/rag.py   -> embeds each dish (mainly name + description + tags)
                        into ChromaDB for semantic search.
    - tools/menu_tools.py -> filters / formats results for the Menu Agent.
    - data/orders.py -> each order references a dish via `item_id`.

Schema of each dish (dict):
    id            (str)        Unique dish ID, e.g. "M001". Orders link to this.
    name          (str)        Display name of the dish.
    category      (str)        Course type: Pizza, Pasta, Main Course, Bowl,
                               Burger, Starter, Beverage.
    cuisine       (str)        Italian, Indian, Fusion, American.
    price         (int)        Price in INR (whole rupees).
    rating        (float)      Average customer rating out of 5.0.
    dietary_tags  (list[str])  Any of "Veg", "Vegan", "GF" (gluten-free).
                               Empty list = no dietary label.
    description   (str)        Short text description. This is the field that
                               semantic search matches most strongly, so keep
                               it descriptive.
    availability  (bool)       True if the dish can be ordered right now.
"""

MENU: list[dict] = [
    {
        "id": "M001",
        "name": "Margherita Pizza",
        "category": "Pizza",
        "cuisine": "Italian",
        "price": 299,
        "rating": 4.7,
        "dietary_tags": ["Veg"],
        "description": "Classic thin crust with tomato, mozzarella, basil",
        "availability": True,
    },
    {
        "id": "M002",
        "name": "Vegan Pasta Primavera",
        "category": "Pasta",
        "cuisine": "Italian",
        "price": 349,
        "rating": 4.5,
        "dietary_tags": ["Vegan"],
        "description": "Penne with seasonal vegetables, olive oil, garlic",
        "availability": True,
    },
    {
        "id": "M003",
        "name": "Butter Chicken",
        "category": "Main Course",
        "cuisine": "Indian",
        "price": 379,
        "rating": 4.9,
        "dietary_tags": ["GF"],
        "description": "Creamy tomato curry with tender chicken and naan",
        "availability": True,
    },
    {
        "id": "M004",
        "name": "Vegan Buddha Bowl",
        "category": "Bowl",
        "cuisine": "Fusion",
        "price": 319,
        "rating": 4.6,
        "dietary_tags": ["Vegan", "GF"],
        "description": "Quinoa, chickpeas, avocado, greens, tahini",
        "availability": True,
    },
    {
        "id": "M005",
        "name": "Classic Cheeseburger",
        "category": "Burger",
        "cuisine": "American",
        "price": 259,
        "rating": 4.4,
        "dietary_tags": [],  # "None" in the spec -> no dietary labels
        "description": "Beef patty, cheddar, lettuce, tomato, brioche bun",
        "availability": True,
    },
    {
        "id": "M006",
        "name": "Paneer Tikka",
        "category": "Starter",
        "cuisine": "Indian",
        "price": 199,
        "rating": 4.8,
        "dietary_tags": ["Veg", "GF"],
        "description": "Tandoor-grilled cottage cheese with peppers",
        "availability": True,
    },
    {
        "id": "M007",
        "name": "Aglio e Olio",
        "category": "Pasta",
        "cuisine": "Italian",
        "price": 279,
        "rating": 4.5,
        "dietary_tags": ["Vegan"],
        "description": "Spaghetti with garlic, chilli, olive oil, parsley",
        "availability": True,
    },
    {
        "id": "M008",
        "name": "Mango Lassi",
        "category": "Beverage",
        "cuisine": "Indian",
        "price": 99,
        "rating": 4.7,
        "dietary_tags": ["Veg", "GF"],
        "description": "Blended yogurt with Alphonso mango, cardamom",
        "availability": True,
    },
]
