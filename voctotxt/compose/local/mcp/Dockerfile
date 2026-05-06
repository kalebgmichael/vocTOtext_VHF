FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir "mcp[cli]>=1.9" "httpx>=0.27"

# mcp_server lives one level above voctotxt (repo root)
COPY mcp_server/ ./mcp_server/

# Default: stdio for Docker MCP gateway (docker mcp server)
# Override MCP_TRANSPORT=http for docker-compose HTTP service
ENV MCP_TRANSPORT=stdio
ENV MCP_PORT=8001
# host.docker.internal reaches the Mac host from inside any Docker container
ENV UPLOAD_URL=http://host.docker.internal:8000/api/transcribe/upload/

EXPOSE 8001

CMD ["python", "mcp_server/transcribe_tool.py"]
