from snackstack.data import ORDERS
from langchain_core.tools import tool
from snackstack.logger import setup_logger
import re


logger = setup_logger("order_tools")

def normalise_order_id(raw: str) -> str:
    """Accept 'ORD101', 'ORD-101', 'ord-101', or just '101' → 'ORD101'."""
    upper = raw.upper().strip()
    clean = upper.replace("ORD-", "").replace("ORD", "").strip()
    return f"ORD-{clean}"

def find_lookup_key(key: str) -> str | None:
    """Return the order_id matching an order ID, tracking ID or email."""
    key = key.strip()
    order_id = normalise_order_id(key)
    if order_id in ORDERS:
        return order_id

    for order_id, order in ORDERS.items():
        if order["tracking_id"].upper() == key.upper():
            return order_id
        if order["customer_email"].lower() == key.lower():
            return order_id
    return None


@tool
def get_order_status(identifier: str) -> str:
    """ Look up an order by order ID (ORD-201), tracking ID (SS201TRK) or customer email.
    Args:
        identifier: str:  order id or tracking id
    Return : 
        dict: A dictionary containing the order status.
    """
    try :
        oid = find_lookup_key(identifier)
        match_order = ORDERS.get(oid) if oid else None

        if not match_order:
            return f"Order {oid} not found. Please verify the order ID."

        
        info = (
            f"Order {oid}:\n"
            f"  Customer : {match_order['customer_name']} ({match_order['customer_email']})\n"
            f"  Product  : {match_order['item_name']}\n"
            f"  Price    : ₹{match_order['price']:,}\n"
            f"  Ordered  : {match_order['order_date']}\n"
            f"  ETA      : {match_order['estimated_delivery']}"
        )

        if match_order.get("status"):
            info += f"\n  Status    : {match_order['status']}"
        return info
    except Exception as e:
        logger.error(f"Error getting order status: {e}")
        return f"Error getting order status: {e}"




##################### TEST the tools  ############################### 

# print(get_order_status.invoke({"identifier": "Where is ORD-201?"}))
# print(get_order_status.invoke({"identifier": "Status of SS201TRK"}))
# print(get_order_status.invoke({"identifier": "priya@example.com"}))
