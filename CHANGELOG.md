# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.2.0-alpha.1
This version introduces breaking changes. Existing code might better stick to [v0.1.2](https://github.com/haowen-xu/tfsnippet/tree/v0.1.2)

### Added
- Utilities have been exported to the root package, and now it's recommended to use TFSnippet by ``import tfsnippet as spt``.
- `tfsnippet.layers` package, including dense layer, convolutional layers, normalization layers, and flow layers.
- `tfsnippet.utils.debugging` module, including several utilities to write debugging code with a global switch to enable/disable.

### Changed
- `global_reuse`, `instance_reuse`, `reopen_variable_scope`, `root_variable_scope` and `VarScopeObject` have been rewritten, and their behaviors have been slightly changed.  This might cause existing code to be malfunction, if these code relies heavily on the precise variable scope or name scope of certain variables or tensors.
- `global_reuse`, `instance_reuse`, `VarScopeObject` have been moved to `tfsnippet.reuse`.
- `Trainer` now accepts `summaries` argument on construction.
- `tfsnippet.flows` now moved to `tfsnippet.layers.flows`, and all its contents
  can be directly found under `tfsnippet.layers` namespace.  The interface of flows has been re-designed.
- Some utilities in `tfsnippet.utils` have been migrated to `tfsnippet.ops`.

### Removed
- The `modules` package has been purged out of this project totally, including the `VAE` class.
- `tfsnippet.mathops` package has been removed.  Some of its members have been migrated to `tfsnippet.ops`.
- `auto_reuse_variables` has been removed.
