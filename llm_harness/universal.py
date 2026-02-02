from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from rich import print

from .llm import ChatOpenRouter
from .tools import get_tools
from .utils import load_image_base64

load_dotenv()


class UniversalChain:
    """A chain implementation using LangChain's create_agent."""

    def __init__(self, model: str | BaseChatModel | None = None):
        """Initialize the UniversalChain with a language model.

        Args:
            model: BaseChatModel instance or model string for ChatOpenRouter
        """
        if isinstance(model, str):
            llm = ChatOpenRouter(model=model)
        elif isinstance(model, BaseChatModel):
            llm = model
        else:
            llm = ChatOpenRouter(model="google/gemini-2.5-flash")

        self.agent = create_agent(
            model=llm,
            tools=get_tools(),
            checkpointer=MemorySaver(),
        )
        self.thread_id = "universal-chain-session"

    def invoke(self, text: str, image: str | None = None) -> dict:
        """Invoke and get the response from the agent.

        Args:
            text: The input text
            image: Path or URL to an image to include

        Returns:
            Dictionary containing the agent response
        """
        message = _create_message(text, image)
        config = {"configurable": {"thread_id": self.thread_id}}
        return self.agent.invoke({"messages": [message]}, config=config)

    def invoke_as_str(self, text: str, image: str | None = None) -> str:
        """Invoke and get the string response from the agent.

        Args:
            text: The input text
            image: Path or URL to an image to include

        Returns:
            The response string
        """
        response = self.invoke(text, image)
        return response["messages"][-1].content


def _create_message(text: str, image: str | None = None) -> HumanMessage:
    """Create a human message with text and optional image content.

    Args:
        text: The text input from the user
        image: Path or URL to an image to include

    Returns:
        HumanMessage: A formatted message object containing text and optional image
    """
    content = []
    if text:
        content.append({"type": "text", "text": text})
    if image:
        try:
            image_base64 = load_image_base64(image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )
        except Exception as e:
            print(f"Error loading image: {e}")
            print("Skipping image")
    return HumanMessage(content=content)
