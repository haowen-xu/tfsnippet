# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.2.0-alpha.4
This version introduces breaking changes. Existing code might better stick to [v0.1.2](https://github.com/haowen-xu/tfsnippet/tree/v0.1.2)

### Added
- Utilities have been exported to the root package, and now it's recommended to use TFSnippet by ``import tfsnippet as spt``.
- Added `layers` package, including dense layer, convolutional layers, normalization layers, and flow layers.
- Added a class `Config` to define user configs.
- Added the global config object `settings`.
- Added `model_variable` and `get_model_variables`; now all layer variables are created via `model_variable` function, instead of `tf.get_variable`.
- Added `CheckpointSaver`.
- Added `utils.EventSource`.

### Changed
- Pin `ZhuSuan` dependency to the last commit (48c0f4e) of 3.x.
- `global_reuse`, `instance_reuse`, `reopen_variable_scope`, `root_variable_scope` and `VarScopeObject` have been rewritten, and their behaviors have been slightly changed.  This might cause existing code to be malfunction, if these code relies heavily on the precise variable scope or name scope of certain variables or tensors.
- `Trainer` now accepts `summaries` argument on construction.
- `flows` package now moved to `layers.flows`, and all its contents
  can be directly found under `layers` namespace.  The interface of flows has been re-designed.
- Some utilities in `utils` have been migrated to `ops`.
- Added `ScheduledVariable` and `AnnealingScheduledVariable`, to replace `SimpleDynamicValue` and `AnnealingDynamicValue`.  `DynamicValue` is still reserved.
- `BayesianNet.add` now removes the `flow` argument.
- The hook facility of `BaseTrainer` and `Evaluator` have been rewritten with `utils.EventSource`.
- `TrainLoop` now supports to make checkpoints, and recover from the checkpoints.
- Several utilities of `utils.shape_utils` and `utils.type_utils` have been moved from `utils` package to `ops` package.

### Removed
- The `modules` package has been purged out of this project totally, including the `VAE` class.
- `mathops` package has been removed.  Some of its members have been migrated to `ops`.
- `auto_reuse_variables` has been removed.
- `VariableSaver` has been removed.
- `EarlyStopping` has been removed.
- `VariationalTrainingObjectives.rws_wake` has been removed.
