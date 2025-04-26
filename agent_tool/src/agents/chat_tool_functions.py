from datetime import timedelta

from pydantic import BaseModel
from restack_ai.agent import (
    NonRetryableError,
    agent,
    import_functions,
    log,
)

with import_functions():
    from openai import pydantic_function_tool

    from src.functions.llm_chat import (
        LlmChatInput,
        Message,
        llm_chat,
    )
    from src.functions.lookup_sales import (
        LookupSalesInput,
        lookup_sales,
    )


class MessagesEvent(BaseModel):
    messages: list[Message]


class EndEvent(BaseModel):
    end: bool


@agent.defn()
class AgentChatToolFunctions:
    def __init__(self) -> None:
        self.end = False
        self.messages = [Message(
            role="system",
            content="""You are Tinkerbell, a cheerful assistant from Renasanz Salon, located at Grunberger Strasse 24, Berlin. 
Analyze the user's request carefully. If you can assist, respond warmly and directly, reflecting the friendly spirit of the salon. 
If the request is too complex or needs a human touch, kindly reply with 'forward to human'."""
        )]

    @agent.event
    async def messages(self, messages_event: MessagesEvent) -> list[Message]:
        log.info(f"Received messages: {messages_event.messages}")
        self.messages.extend(messages_event.messages)

        tools = [
            pydantic_function_tool(
                model=LookupSalesInput,
                name=lookup_sales.__name__,
                description="Lookup sales for a given category",
            ),
        ]

        try:
            completion = await agent.step(
                function=llm_chat,
                function_input=LlmChatInput(
                    messages=self.messages, tools=tools
                ),
                start_to_close_timeout=timedelta(seconds=120),
            )
        except Exception as e:
            error_message = f"Error during llm_chat: {e}"
            raise NonRetryableError(error_message) from e
        else:
            log.info(f"completion: {completion}")

            assistant_content = completion.choices[0].message.content or ""
            tool_calls = completion.choices[0].message.tool_calls

            # Check if the assistant wants to forward to human
            if "forward to human" in assistant_content.lower():
                self.messages.append(
                    Message(
                        role="assistant",
                        content="Thank you for your request. An operator will reach out to you.",
                    )
                )
            else:
                self.messages.append(
                    Message(
                        role="assistant",
                        content=assistant_content,
                        tool_calls=tool_calls,
                    )
                )

                if tool_calls:
                    for tool_call in tool_calls:
                        log.info(f"tool_call: {tool_call}")

                        name = tool_call.function.name

                        match name:
                            case lookup_sales.__name__:
                                args = LookupSalesInput.model_validate_json(
                                    tool_call.function.arguments
                                )

                                log.info(f"calling {name} with args: {args}")

                                try:
                                    result = await agent.step(
                                        function=lookup_sales,
                                        function_input=LookupSalesInput(category=args.category),
                                        start_to_close_timeout=timedelta(seconds=120),
                                    )
                                except Exception as e:
                                    error_message = f"Error during lookup_sales: {e}"
                                    raise NonRetryableError(error_message) from e
                                else:
                                    self.messages.append(
                                        Message(
                                            role="tool",
                                            tool_call_id=tool_call.id,
                                            content=str(result),
                                        )
                                    )

                                    try:
                                        completion_with_tool_call = await agent.step(
                                            function=llm_chat,
                                            function_input=LlmChatInput(
                                                messages=self.messages
                                            ),
                                            start_to_close_timeout=timedelta(seconds=120),
                                        )
                                    except Exception as e:
                                        error_message = f"Error during llm_chat: {e}"
                                        raise NonRetryableError(error_message) from e
                                    else:
                                        self.messages.append(
                                            Message(
                                                role="assistant",
                                                content=completion_with_tool_call.choices[
                                                    0
                                                ].message.content
                                                or "",
                                            )
                                        )

            return self.messages

    @agent.event
    async def end(self) -> EndEvent:
        log.info("Received end")
        self.end = True
        return {"end": True}

    @agent.run
    async def run(self, agent_input: dict) -> None:
        log.info("AgentChatToolFunctions agent_input", agent_input=agent_input)
        await agent.condition(lambda: self.end)
