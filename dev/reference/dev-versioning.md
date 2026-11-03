# Toolbox versioning policy

Last updated: 2026-10-21

The toolbox uses manual Semantic Versioning. Only an explicit owner release
request opens release preparation; ordinary commits and milestone seals do not
create releases or bump a version.

## Authority and compatibility

Git release tags `vX.Y.Z` are the authoritative toolbox version. Milestone tags
are development checkpoints. Existing `v0.1.0-alpha` and `v0.2.0-alpha` tags
remain historical releases; future release tags use plain `vX.Y.Z`. Do not
rewrite historical tags or infer a released version from the current branch.
The runner's opening `toolbox` row uses `vX.Y.Z (<short-commit>)`: the nearest
ancestral release tag plus the current commit, also recorded in the invocation JSON.
That tag does not version subsequent changes; the commit identifies the checkout.
Append `local changes` for a modified checkout. Without a release tag, display
`unversioned (<short-commit>)`; without a commit, display `version unavailable`.
An unknown working-tree status is marked explicitly.

Compatibility covers documented user commands, configuration keys and meaning,
durable output paths and formats, and scientific calculation methods. Internal
refactoring and disposable intermediate files are not supported interfaces.
Before 1.0 these contracts may change, but every incompatible release must
identify the break and give migration guidance.

| Change | Before 1.0 | From 1.0 onward |
|---|---|---|
| Compatible fix | Patch | Patch |
| Compatible feature | Minor | Minor |
| Incompatible interface or scientific-method change | Minor, with breaking-change notice | Major |

Classify the complete release since the preceding release tag using its largest
required bump. A milestone number or workflow revision is not a toolbox version.
Version `1.0.0` requires explicit owner acceptance of interface stability;
neither test coverage nor a milestone seal establishes stability. Later major
releases also require explicit owner approval.

A numerical bug fix may be a patch when it restores the documented method.
Document which results change, why, and their measured magnitude and scope;
validate against appropriate reference results. A deliberate change to the
scientific method is incompatible even when commands and file schemas stay the
same. Do not re-record a baseline merely to make a gate pass: explain and accept
the result change first, retaining the evidence needed to interpret it.

## Release checklist

1. Obtain the explicit release request. Compare the candidate with the previous
   release tag, reconcile `CHANGELOG.md`'s Unreleased section, and classify the
   whole release. Identify compatibility breaks, migration guidance, numerical
   changes, and known limitations. Agree the target version; obtain explicit
   stability or stable-major approval when applicable.
2. Prepare the dated changelog release entry on the task branch and leave an
   empty Unreleased section for subsequent work. Follow the existing branch
   validation and landing gates in `AGENTS.md` and
   [the validation ladder](dev-validation.md). Landing requires approval.
3. Verify coverage of the candidate main commit and the release delta using
   the validation ladder, including its pre-publication batch gate. Local landing
   alone does not establish release readiness. Reuse applicable landing, CI and execution evidence;
   releasing or tagging the same content does not require another test run.
   Select software, execution, numerical, and visual coverage through that reference;
   it also owns CI limitations and baseline prerequisites. Add `semantic_tree_diff.py`
   when claiming tree compatibility after shape changes.
   Retain the commands, results and candidate/input identities with the evidence.
4. With explicit tagging authorization, create an annotated `vX.Y.Z` tag on
   the verified main commit. Confirm the tag is new and points to that exact
   commit. Never tag a task branch as a toolbox release.
5. Obtain separate authorization to push main and the release tag. Before the
   main push, satisfy the local push gate; after it, read the CI run it triggered
   and confirm both platforms pass. A tag alone is not covered by this repo's
   branch-filtered CI. Resolve failed CI before publishing.
6. Publish a GitHub release or other distribution only with separate owner
   authorization, using the approved tag and release notes. A local tag, a push,
   and publication are distinct actions.

This checklist defines preparation; it does not authorize any release action.
Do not automatically replace a tag after a failure or rewrite a published
release. Preserve the evidence and obtain an owner decision; correct published
versions forward.

## Tooling boundary

The current `git-workflow` versioning helper supports `brain-skill`, `brain-role`,
and generic `command` adapters. Each requires a repository-relative file
carrier and apply/validate actions; it has no authoritative Git-tag adapter.
Its shared SemVer parser also excludes the historical alpha syntax.

Accordingly, `.git-workflow.yml` intentionally omits `versioning:`. This policy
is enforced through the documented manual workflow, not that helper. Do not
invent a `VERSION` file, package version, command adapter, or release automation
to satisfy a file-carrier schema: that would create a second authority. Revisit
configuration only if a tag-based adapter becomes available and is adopted
explicitly. No release is created by adopting this policy.
