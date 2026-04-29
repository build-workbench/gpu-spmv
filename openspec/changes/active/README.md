#OpenSpec Changes Active Directory

This directory contains active change proposals for the GPU SpMV project.

## Purpose

When proposing changes to the codebase:

1. **Create a proposal**: Copy `proposal-template.md` and name it appropriately
2. **Describe the change**: Fill in the template with your proposed changes
3. **Get review**: Discuss with maintainers
4. **Implement**: After approval, implement the changes
5. **Archive**: Move to `../archive/` when complete

## Workflow

```
openspec/changes/
├── active/              # ← Current work (you are here)
│   ├── README.md        # This file
│   └── proposal-template.md
└── archive/             # Completed changes
    ├── 2025-01-15-csr-format/
    ├── 2025-02-10-ell-format/
    └── ...
```

## Creating a Proposal

```bash
#Copy template
cp openspec/changes/active/proposal-template.md openspec/changes/active/YYYY-MM-DD-brief-description.md

#Edit and fill in details
#Submit for review via PR
```

## Related

- Spec directory: `openspec/specs/`
- Project config: `openspec/config.yaml`
