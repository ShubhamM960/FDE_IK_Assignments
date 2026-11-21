from snackstack.tools.rag import retriever
from langchain_core.tools import tool
from snackstack.logger import setup_logger

logger = setup_logger("menu_tools")

@tool
def search_menu_catalog(query: str) -> str:
    """ Search the menu catalog for dishes matching the query (cuisine, diet, price, ingredients).

    Args:
        query: str: What the customer is looking for, e.g. "vegan dishes" or "Indian food under 300".

    Returns:
        str: The matching menu items (name, cuisine, price, dietary tags, description).
    """
    try :
        docs  = retriever.invoke(query)
        if not docs:  # FIX: `result` was undefined -> NameError
            return "No menu items found matching the query."

        logger.info(f"for query {query} :  {len(docs)} menu items found matching the query")
        results = ""  # FIX: `results` was used before assignment
        for i, doc in enumerate(docs, 1):
            results += f"Menu item {i}:\n{doc.page_content}\n\n"
        return results
    except Exception as e:
        logger.error(f"Error searching menu catalog: {e}")
        return f"Error searching menu catalog: {e}"

