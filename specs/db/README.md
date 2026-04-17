# Database Schema Specifications

This directory would contain database schema specifications if the project uses persistent storage.

Currently, the GPU SpMV project is a **compute library** that does not use database storage, so this directory is intentionally left empty.

## When to Add Specs Here

Add database schema specifications to this directory if future versions of the project include:
- Result caching to disk
- Persistent benchmark storage
- Configuration database
- Result analytics

## Spec Format

If database specs are needed, they should follow the conventions in `/specs/README.md`:
- Use `.md` files for human-readable specs
- Include version, status, and last-updated date
- Reference related product requirements and RFCs

---

**Status**: ⏸️ Not applicable (no database storage)
**Last Updated**: 2025-04-17
