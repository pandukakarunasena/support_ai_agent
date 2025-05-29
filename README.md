docker run -d \
  --env-file .env.run \
  -v hf_cache:/home/appuser/.cache/huggingface \
  -p 9999:9999 \
  u2-git-mcp-server:latest

docker network create mynet

docker run --rm -it \
  --env-file .env.run \
  -v hf_cache:/home/appuser/.cache/huggingface \
  -p 9999:9999 \
  --name u2-git-mcp_server \
  --network mynet \
  u2-git-mcp-server:latest

