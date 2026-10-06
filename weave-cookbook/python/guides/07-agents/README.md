# Track agents

This optional, machine-level integration traces an agent you *use* rather than code in this repository. The [Claude Code plugin](https://docs.wandb.ai/weave/guides/integrations/agents/claude-code-harness) logs every session — turns, tool calls, subagents, tokens, latency — with no code changes.

> The plugin sends session data to Weave: prompts, responses, tool inputs and outputs, file contents, shell output. There is no PII scrubbing. If that conflicts with your security requirements, don't install it.

## Track Claude Code

These commands change your machine, not this repository: they install a global npm package (`weave-claude-code`, v0.2.13 at capture time) and register a plugin plus a background daemon in Claude Code.

```bash
npm install -g weave-claude-code
weave-claude-code install
claude
```

The installer prompts for your Weave project and API key (or reads `WEAVE_PROJECT` and `WANDB_API_KEY` with `--non-interactive`). From then on, every Claude Code session traces automatically. `weave-claude-code status` checks the setup; `weave-claude-code logs --follow` tails the daemon; `weave-claude-code uninstall` removes it.

## Inspect it in Weave

Open your project, select **Agents**, then the **Conversations** tab. Each session is one conversation; each user prompt is one trace:

```text
invoke_agent claude-code
├─ chat <model>
├─ execute_tool <tool_name>        Read, Bash, Grep, ...
└─ invoke_agent <subagent_type>
   ├─ chat <model>
   └─ execute_tool <tool_name>
```

This is the same span vocabulary from [guide 01](../01-tracing/README.md) — `invoke_agent`, `chat`, `execute_tool` — emitted by the harness instead of your code. The agent name defaults to `claude-code`; change it with `weave-claude-code config set agent_name <name>`.

## Other integrations

| Kind | Integrations | How they attach |
| --- | --- | --- |
| Harness plugins | [Codex](https://docs.wandb.ai/weave/guides/integrations/agents/codex-harness), [OpenClaw](https://docs.wandb.ai/weave/guides/integrations/agents/openclaw-harness), [Pi](https://docs.wandb.ai/weave/guides/integrations/agents/pi-dev-harness) | Install the plugin or extension, then use the tool |
| Agent SDKs | [OpenAI Agents SDK](https://docs.wandb.ai/weave/guides/integrations/agents/openai-agents-sdk), [Google ADK](https://docs.wandb.ai/weave/guides/integrations/agents/google-adk), [Claude Agent SDK](https://docs.wandb.ai/weave/guides/integrations/agents/claude-agents-sdk) | `weave.init(...)` in your script autopatches the SDK |
| Custom loops | [Guide 01](../01-tracing/README.md), or [OTel spans](../08-otel/README.md) from any pipeline | Weave span helpers, or plain OpenTelemetry |

The full list lives in [Choose an agent integration](https://docs.wandb.ai/weave/agent-integration-quickstart).

This guide's commands mirror the official plugin docs but aren't executed by this repository's checks — the plugin installs globally and traces your personal Claude Code sessions, so run it deliberately.
