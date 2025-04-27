from datetime import timedelta

from pydantic import BaseModel
from restack_ai.agent import (
    NonRetryableError,
    agent,
    import_functions,
    log,
)

with import_functions():
    from src.functions.llm_chat import (
        LlmChatInput,
        Message,
        llm_chat,
    )


class MessagesEvent(BaseModel):
    messages: list[Message]


class EndEvent(BaseModel):
    end: bool


@agent.defn()
class AgentChatToolFunctions:
    def __init__(self) -> None:
        self.end = False
        self.messages = [
            Message(
                role="system",
                content="""You are Tinkerbell, a cheerful assistant from Renasanz Salon, located at Grunberger Strasse 24, Berlin. 
                        Analyze the user's request carefully. 

                        Open hours: 10AM-6PM

                        If you can assist, respond warmly and directly, reflecting the friendly spirit of the salon. 

                        If the user is asking for booking an appointment
                        After asking info about the appointment reply with 'forward to human'.

                        If the request is too complex or needs a human touch, kindly reply with 'forward to human'."""
            )
        ]

    @agent.event
    async def messages(self, messages_event: MessagesEvent) -> list[Message]:
        log.info(f"Received messages: {messages_event.messages}")
        self.messages.extend(messages_event.messages)

        try:
            completion = await agent.step(
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
            log.info(f"completion: {completion}")

            assistant_content = completion.choices[0].message.content or ""


            # Check if the assistant wants to forward to human
            if "forward to human" in assistant_content.lower():
                self.messages.append(
                    Message(
                        role="assistant",
                        content="Thank you for your request. An operator will reach out to you.",
                    )
                )
                # Add a system message with the task summary for the operator
                self.messages.append(
                    Message(
                        role="system",
                        content=f"operator_task: {assistant_content}",
                    )
                )
            else:
                self.messages.append(
                    Message(
                        role="assistant",
                        content=assistant_content,
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
