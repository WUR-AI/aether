# Contributing

Thank you very much for contributing to the AETHER! Below are some guidelines to ensure we can integrate new contributions effectively:

When contributing to this repository, please first discuss the change you wish to make, preferably via an Issue, with the administrators of this repository before making a change.

In general, all contributions should be made via Pull Requests.

Please note we have a [code of conduct](https://github.com/WUR-AI/aether/blob/main/CODE_OF_CONDUCT.md), please follow it in all your interactions with the project.

## Pull Request Process

1. Create a new feature branch from `develop`, called `feature/my_feature`. Implement your changes here. When you're done, create a PR with `develop`. (If `develop` has been updated in the meanwhile, merge `develop` into your feature branch `feature/my_feature` first to get the updates).
2. Before creating a PR, 1) add and/or update any tests if necessary and 2) run `pytest` locally (`pytest -v --use-mock` in CLI) to ensure compatibility with all existing tests.
3. Create a PR to merge into `develop`.
4. Please link a PR to the open issue it resolves, if relevant.
5. All PRs must be reviewed by one of the project admins before being merged. Please assign @vdplasthijs or @gabrieletijunaityte to review the code.

## Attribution

Adapted from https://gist.github.com/PurpleBooth/b24679402957c63ec426
