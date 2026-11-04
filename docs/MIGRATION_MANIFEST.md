# Migration provenance and current repository map

The original migration manifest is preserved byte-for-byte at
[legacy_files/historical_reports/docs/MIGRATION_MANIFEST.md](../legacy_files/historical_reports/docs/MIGRATION_MANIFEST.md).
Its recorded checksums, original paths, Phase 0 findings and restoration context
have not been rewritten to describe later recognition work.

Current navigation:

- [Repository file migration map](reproducibility/FILE_MIGRATION_MAP.md): significant
  move groups, reasons and a complete mapping of all 710 original tracked files.
- [Repository hygiene report](reproducibility/REPOSITORY_HYGIENE_REPORT.md): source
  isolation, unchanged scientific interfaces, validation and remaining limitations.
- [Historical archive](../legacy_files/README.md): preserved implementations,
  annotations, images, diagnostics and dated reports.
- [Current restoration guide](CLUSTER_RESTORE.md): active environment instructions
  and the distinction between current and historical layouts.
- [Reproducibility resources](../reproducibility/): unchanged dated manifests,
  upstream revisions/patches, environment snapshots and experimental evidence.

The isolated reorganization starts from production commit
`fcb0a40db35c1c20da76df5fa90fa7a9179d63ad`. The production branch and remote cluster
checkout remain unchanged. Existing B0/B1 source, configurations, seeds,
preprocessing, labels and checkpoint formats retain their original paths and
bytes. This hygiene work makes no new scientific performance claim.
