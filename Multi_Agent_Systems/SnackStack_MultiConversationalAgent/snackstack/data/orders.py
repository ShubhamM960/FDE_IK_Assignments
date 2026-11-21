"""
SnackStack Order Database
=========================

A mock order database, stored as a dict keyed by `order_id` so that a lookup
by Order ID is a direct O(1) access: ORDERS["ORD-201"].

How it's used later:
    - tools/order_tools.py -> `get_order_status` looks up an order by
      order_id (direct key), tracking_id or customer_email (scan the values).

Schema of each order (dict):
    order_id            (str)  Unique order ID, e.g. "ORD-201". Also the dict key.
    item_id             (str)  ID of the ordered dish -> matches MENU[i]["id"].
    item_name           (str)  Dish name, stored here so we don't have to
                               join with the menu just to display it.
    customer_name       (str)  Name of the customer who placed the order.
    customer_email      (str)  Customer email (another way to look up orders).
    status              (str)  Lifecycle stage. One of:
                               "Placed" -> "Preparing" -> "Out for Delivery"
                               -> "Delivered"
    price               (int)  Amount paid in INR (same as the dish price).
    order_date          (str)  When the order was placed, "YYYY-MM-DD HH:MM".
    estimated_delivery  (str)  Expected (or actual, if delivered) delivery
                               time, "YYYY-MM-DD HH:MM".
    tracking_id         (str)  Delivery tracking code, e.g. "SS201TRK".
"""

ORDERS: dict[str, dict] = {
    "ORD-201": {
        "order_id": "ORD-201",
        "item_id": "M003",
        "item_name": "Butter Chicken",
        "customer_name": "Priya Nair",
        "customer_email": "priya@example.com",
        "status": "Out for Delivery",
        "price": 379,
        "order_date": "2026-11-20 12:15",
        "estimated_delivery": "2026-11-20 13:00",
        "tracking_id": "SS201TRK",
    },
    "ORD-202": {
        "order_id": "ORD-202",
        "item_id": "M001",
        "item_name": "Margherita Pizza",
        "customer_name": "Arjun Mehta",
        "customer_email": "arjun@example.com",
        "status": "Placed",
        "price": 299,
        "order_date": "2026-11-20 16:45",
        "estimated_delivery": "2026-11-20 17:30",
        "tracking_id": "SS202TRK",
    },
    "ORD-203": {
        "order_id": "ORD-203",
        "item_id": "M005",
        "item_name": "Classic Cheeseburger",
        "customer_name": "Sneha Roy",
        "customer_email": "sneha@example.com",
        "status": "Preparing",
        "price": 259,
        "order_date": "2026-11-20 16:20",
        "estimated_delivery": "2026-11-20 17:05",
        "tracking_id": "SS203TRK",
    },
    "ORD-204": {
        "order_id": "ORD-204",
        "item_id": "M004",
        "item_name": "Vegan Buddha Bowl",
        "customer_name": "Rahul Das",
        "customer_email": "rahul@example.com",
        "status": "Delivered",
        "price": 319,
        "order_date": "2026-11-19 19:30",
        "estimated_delivery": "2026-11-19 20:10",
        "tracking_id": "SS204TRK",
    },
    "ORD-205": {
        "order_id": "ORD-205",
        "item_id": "M006",
        "item_name": "Paneer Tikka",
        "customer_name": "Kavya Sharma",
        "customer_email": "kavya@example.com",
        "status": "Placed",
        "price": 199,
        "order_date": "2026-11-20 16:55",
        "estimated_delivery": "2026-11-20 17:40",
        "tracking_id": "SS205TRK",
    },
}
