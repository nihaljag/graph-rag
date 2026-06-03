# Prompts

By default, specgraph uses GraphRAG's **built-in** extraction prompts. This is the
most robust choice (the prompt templates always match the installed GraphRAG
version's expected placeholders), and the highest-leverage tuning — the
**entity types** and the **claim/covariate description** — is already driven from
`config.yaml` (`graphrag.entity_types`, `graphrag.domain_entity_types`,
`graphrag.claims_description`).

## Optional: custom, behavior-aware prompts (advanced)

If you want to push relationship recall further (e.g. explicitly instruct the
model to extract *references / affects / uses-structure / produces-error /
returns-status* relations with canonical entity names), generate the default
prompt files for your installed version and edit them:

```bash
# In a scratch dir, on a machine with the same graphrag version:
graphrag init --root ./_scratch
# Copy ./_scratch/prompts/*.txt here and edit extract_graph.txt etc.
```

Then point GraphRAG at them by adding `prompt:` keys under the relevant workflow
sections in the rendered `graphrag_workspace/settings.yaml` (or extend
`src/specgraph/index/settings_render.py` to set them). Keep edits
specification-agnostic — describe *kinds* of relationships, not NVMe specifics.
