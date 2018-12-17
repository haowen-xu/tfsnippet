# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.2.0-alpha.1
This version introduces breaking changes. Existing code might better stick to [v0.1.2](https://github.com/haowen-xu/tfsnippet/tree/v0.1.2)

### Changed
- `global_reuse` the `VarScopeObject` is now rewritten.
  This will cause existing code to be malfunction, if they rely heavily on the precise variable scope or name scope of certain variables or tensors.

### Removed
- The `modules` package is now purged out of this project totally, including the `tfsnippet.modules.VAE` class.
- `instance_reuse` has been removed, and will not be rewritten until we have got enough evidence of its necessity.
  The recommended way to replace `instance_reuse` is to use `tf.make_template` inside `VarScopeObject._variable_scope_created`,
  to wrap some of your own methods into template methods.
