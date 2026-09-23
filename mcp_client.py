

from mcp.client.client import types

class McpClient():
    async def list_tools(self) -> list[types.Tool]:
        result = await self.session().list_tools()
        return result.tools


    async def call_tool(
        self, tool_name: str, tool_input: dict
    ) -> types.CallToolResult | None:
        return await self.session().call_tool(tool_name, tool_input)
