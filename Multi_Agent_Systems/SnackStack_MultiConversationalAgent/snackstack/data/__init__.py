"""
SnackStack data package.

Exposes the mock data sources so the rest of the app can import them from
one place:

    from snackstack.data import MENU, ORDERS
"""

from .menu import MENU
from .orders import ORDERS

# Controls what `from snackstack.data import *` exposes, and documents
# the public API of this package.
__all__ = ["MENU", "ORDERS"]
