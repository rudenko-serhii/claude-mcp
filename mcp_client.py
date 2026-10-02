import asyncio
import sys
import json

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Prompt, TextContent, TextResourceContents

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp_types import GetPromptResult


load_dotenv()

MODEL = "glm-5.3"
anthropic = Anthropic()


async def read_resource(client: Client, uri: str) -> str | list[str]:
    result = await client.read_resource(uri)
    resource = result.contents[0]

    if isinstance(resource, TextResourceContents):
        if resource.mime_type == "application/json":
            return json.loads(resource.text)

    return resource.text


async def list_prompts(client: Client) -> list[Prompt]:
    result = await client.list_prompts()
    return result.prompts


def server_params(server_script_path: str) -> StdioServerParameters:
    """Describe the subprocess that runs an MCP server

    Args:
        server_script_path: Path to the server script (.py or .js)
    """
    if server_script_path.endswith(".py"):
        command = sys.executable
    elif server_script_path.endswith(".js"):
        command = "node"
    else:
        raise ValueError("Server script must be a .py or .js file")

    return StdioServerParameters(command=command, args=[server_script_path])


async def process_query(client: Client, query: str | GetPromptResult) -> str:
    """Process a query using Claude and available tools"""
    if isinstance(query, GetPromptResult):
        messages =  [{"role": m.role, "content": m.content.text} for m in query.messages]
    else: 
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

    tool_list = await client.list_tools()
    available_tools = [{
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema
    } for tool in tool_list.tools]

    response = anthropic.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=messages,
        tools=available_tools
    )

    final_text = []

    need_to_call_the_tool = True


    while need_to_call_the_tool:
        tool_results = []

        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                result = await client.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content.id,
                    "content": "\n".join(
                        block.text
                        for block in result.content
                        if isinstance(block, TextContent)
                    ),
                    "is_error": result.is_error
                })

        if tool_results:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            response = anthropic.messages.create(
                model=MODEL,
                max_tokens=1000,
                messages=messages,
                tools=available_tools
            )

        need_to_call_the_tool = any(c.type == 'tool_use' for c in response.content)

    for content in response.content:
        if content.type == 'text':
            final_text.append(content.text)

    return "\n".join(final_text)

async def handle_prompt(client: Client, query: str) -> str:
    list_prompts_res = await client.list_prompts()

    if (len(query) == 1):
        return("\n" + ', '.join([prompt.name for prompt in list_prompts_res.prompts]))
    else:
        prompt_name, *arguments = query.removeprefix('/').split(' ')

        if len(arguments) == 0:
            return(f"No arguments from prompt {prompt_name}")

        prompt =  next((prompt for prompt in list_prompts_res.prompts if prompt.name == prompt_name), None)

        if not prompt:
            return(f"No such prompt {prompt_name}")

        prompt_response = await client.get_prompt(name=prompt_name, arguments={prompt.arguments[0].name: arguments[0]})
        response = await process_query(client, prompt_response)
        return("\n" + response)

async def chat_loop(client: Client) -> None:
    """Run an interactive chat loop"""

    print("\nMCP Client Started!")
    print("Type your queries or 'quit' to exit.")

    while True:
        try:
            query = (await asyncio.to_thread(input, "\nQuery: ")).strip().lower()
        except EOFError:
            break

        if query == 'quit':
            break

        if not query: continue
        query_command = query[0]

        try:
            match query_command:
                case '@':
                    if (query == query_command):
                        response = await read_resource(client, "docs://documents")
                        print("\n" + ', '.join(response))
                    else:
                        response = await read_resource(client, f"docs://documents/{query.removeprefix('@')}")
                        print("\n" + response)
                case '/':
                    promp_result = await handle_prompt(client, query)
                    print("\n" + promp_result)
                case _:
                    response = await process_query(client, query)
                    print("\n" + response)

        except Exception as e:
            print(f"\nError: {e}")

async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)



    async with Client(stdio_client(server_params(sys.argv[1]))) as client:
        tool_list = await client.list_tools()
        tool_names = [tool.name for tool in tool_list.tools]
        print("\nConnected to server with tools:", tool_names)


        await chat_loop(client)


if __name__ == "__main__":
    asyncio.run(main())