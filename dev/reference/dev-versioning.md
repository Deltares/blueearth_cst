# Toolbox versioning policy

Last updated: 2026-11-05

The toolbox uses manual Semantic Versioning. A release requires an explicit request; ordinary commits and
development milestones do not create one.

## Authority and compatibility

Git release tags `vX.Y.Z` are the version authority; new tags are annotated. Earlier `v0.1.0-alpha` and
`v0.2.0-alpha` tags remain valid historical releases; new tags use plain `vX.Y.Z`. Do not rewrite tags
or infer a release from a branch name. The runner reports the nearest ancestral release tag and current
short commit in its `toolbox` row and invocation JSON. It marks local changes or unknown worktree status;
without a release tag it says `unversioned (<short-commit>)`, and without a commit, `version unavailable`.
The commit identifies the exact checkout; the preceding tag does not version its later changes.

Compatibility covers documented commands, configuration keys and meaning, durable output paths and formats,
and scientific calculation methods. Internal refactors and disposable intermediates are outside that surface.
Before 1.0, incompatible changes still need a breaking-change notice and migration guidance.

| Change | Before 1.0 | From 1.0 onward |
|---|---|---|
| Compatible fix | Patch | Patch |
| Compatible feature | Minor | Minor |
| Incompatible interface or scientific-method change | Minor, with breaking-change notice | Major |

Classify the full delta since the preceding release by its largest required bump. Version `1.0.0` requires
explicit acceptance of interface stability; tests or a milestone alone do not establish it. Later major
releases also require explicit approval.

A numerical bug fix may be a patch if it restores the documented method. Explain the changed results,
cause, measured magnitude and scope, and validate against suitable references. A deliberate method change
is incompatible even when commands and schemas stay stable. Do not re-record a baseline merely to pass a
gate; first explain and accept the result change.

## Release checklist

1. On an explicit release request, compare the candidate with the preceding release tag. Reconcile
   `CHANGELOG.md`'s Unreleased section, classify the full delta, and identify breaks, migration steps,
   numerical changes and known limitations. Agree the target version and any stability decision.
2. Prepare a dated changelog entry and an empty Unreleased section. Follow the
   [Git conventions](dev-git-conventions.md) and [validation ladder](dev-validation.md).
3. Verify the candidate public trunk (`main`) commit and release delta under that ladder, including
   its pre-publication batch gate. Reuse applicable evidence for the same content and inputs; a tag
   alone needs no new run.
   Select software, execution, numerical and visual checks by affected claims. Use
   `semantic_tree_diff.py` when claiming tree compatibility after a shape change. Retain commands,
   results, candidate identity and relevant input identity.
4. With explicit tagging authorization, create a new annotated `vX.Y.Z` tag on the verified `main`
   commit and confirm its target. Never tag a task branch as a toolbox release.
5. Push `main` and the tag only with separate authorization. Satisfy the pre-push gate before pushing
   `main`, then confirm CI passes on both platforms for that commit. A tag push alone does not run the
   branch-filtered CI; resolve failures before publication.
6. Publish a release or distribution only with separate authorization and the approved tag and notes.

Tagging, pushing and publication are distinct actions. Do not replace a tag after failure or rewrite a
published release; preserve evidence and correct published versions forward.

## Tooling boundary

Inspect local release history with `git tag --merged HEAD --list 'v*'`, `git show <tag>`, and
`git diff <tag>..HEAD`; select an actual release tag, since the listing can include other `v*` names.
The tag remains authoritative. Do not add a `VERSION` file, package version, build backend or packaging
tables as a side effect of ordinary tooling changes; `pyproject.toml` currently defines none of these.
Repository Git automation intentionally declares no file-based version carrier because release tags are
authoritative. Its absence does not create another version source or automate a release.
