# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.2.0-alpha.1
This version introduces breaking changes. Existing code might better stick to [v0.1.2](https://github.com/haowen-xu/tfsnippet/tree/v0.1.2)

### Added
- `tfsnippet.utils.debugging` module, including several utilities to write debugging code with a global switch to enable/disable.

### Changed
- `global_reuse`, `instance_reuse`, `reopen_variable_scope`, `root_variable_scope` and `VarScopeObject` have been rewritten, and their behaviors have been slightly changed.
  This might cause existing code to be malfunction, if these code relies heavily on the precise variable scope or name scope of certain variables or tensors.
- `Trainer` now accepts `summaries` argument on construction.

### Removed
- The `modules` package is now purged out of this project totally, including the `VAE` class.
- `auto_reuse_variables` has been removed.
