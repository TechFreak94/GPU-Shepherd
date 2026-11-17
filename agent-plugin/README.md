# GPU Shepherd for AI agents

Let an AI agent look after your GPU Shepherd: see what is running on which GPU, start, stop and
swap apps, read logs, install App Center apps, add models and apply updates. The agent asks you
before every change, and a start that would stop or move other apps always shows you the plan first.

It has two parts:

- **An MCP server built into GPU Shepherd** at `/api/v1/mcp`. It uses the same API token as the
  [JSON API](../docs/api.md) and does exactly what the matching endpoints do. Any agent that speaks
  MCP can use it.
- **A Claude Code plugin** (this folder): connects Claude Code to that server and adds a skill
  ([skills/gpu-shepherd/SKILL.md](skills/gpu-shepherd/SKILL.md)) that explains how GPU Shepherd works
  and the safety rules. Other agents can be given the same file as instructions.

You need the dashboard's address (as you open it in a browser) and its API token, from
**Settings › API and command line**. Whoever has the token can do what the admin can, so keep it
like a password. Across the internet, use an HTTPS address (for example Tailscale's, under
**Settings › Remote access**) so the token isn't sent in the clear.

## Claude Code

```
/plugin marketplace add TechFreak94/GPU-Shepherd
/plugin install gpu-shepherd@gpu-shepherd
```

Claude Code asks for the dashboard address (no slash at the end, e.g. `http://192.168.1.20:8090`)
and the token, and keeps the token in your system's secure storage. Then ask things like *"What's
running on my GPUs?"* or *"Swap ComfyUI out for the big coding model"*.

Without the plugin, add only the MCP server:

```
claude mcp add --transport http gpu-shepherd http://192.168.1.20:8090/api/v1/mcp \
  --header "Authorization: Bearer gs-..."
```

## Other agents

Point the agent's MCP settings at `http://<dashboard>/api/v1/mcp` (Streamable HTTP) with the header
`Authorization: Bearer <token>`, and give it [SKILL.md](skills/gpu-shepherd/SKILL.md) as
instructions. For example, OpenCode's `opencode.json`:

```json
{
  "mcp": {
    "gpu-shepherd": {
      "type": "remote",
      "url": "http://192.168.1.20:8090/api/v1/mcp",
      "headers": { "Authorization": "Bearer {env:GPU_SHEPHERD_TOKEN}" }
    }
  }
}
```

Cursor, Cline, Windsurf and others take the same `url` and `headers` in their MCP settings. A
client that only runs local (stdio) servers can reach it through a bridge such as `mcp-remote`.

## What the agent can and can't do

| Can | Stays in the dashboard |
|---|---|
| GPUs, instances, logs, jobs, App Center, models, disk space, updates (read) | Passwords and the API token |
| Start, stop, restart (plans that affect other apps need your OK) | Reboot and shutdown |
| Call off a change, or stop without waiting for running work | Remote access, tensor unlock |
| Install an app, add or download a model, load or unload one | Deleting instances, models, files; the trash |
| Apply app updates and image updates; update GPU Shepherd | Moving storage, mounting drives, settings |

The tool names and inputs are a stable contract: later versions only add tools and optional inputs.
