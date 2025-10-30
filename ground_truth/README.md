# Ground-truth power model traces

This directory stores JSON files containing the high-quality command traces for
individual problem IDs. Each file should follow the schema produced by the
power model, for example:

```json
{
  "problem_id": "container_kill-analysis-1",
  "key_commands": [
    {
      "command": "exec_shell(\"kubectl get pods -n test-hotel-reservation\")",
      "importance_score": 6,
      "description": "List pods to inspect restart status."
    },
    {
      "command": "submit({\"system_level\": \"Application\", \"fault_type\": \"Dependency Problem\"})",
      "importance_score": 10,
      "type": "execute_command"
    }
  ]
}
```

Problem IDs with multiple variants can either share a single file or have one
file per variant. The RL environment automatically loads every `*.json` file in
this directory when it starts, so remember to keep the data up to date with the
latest power-model outputs.
