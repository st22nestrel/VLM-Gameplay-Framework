# Explicit publishing

`agent-public` has independent history starting with the public runtime snapshot.
Private full-suite history must never become an ancestor of this branch. A
worktree shares the Git repository and its remotes; its separate directory alone
does not isolate history or publishing configuration.

During development, push `agent-public` only to the private repository. When a
version is ready, explicitly push its reviewed commit to public `main`. With
remotes named `private` and `public`, the commands are:

```powershell
git push private agent-public:agent-public
# Run this only when deliberately publishing the reviewed branch tip:
git push public agent-public:main
```

Remote creation and credentials are intentionally not configured by this snapshot.
Use your existing private remote's name if different. There is no automatic
publication hook or workflow.

Every intervening commit on the published branch becomes public, including deleted
files and commit messages. Keep private experiments on separate branches and port
only reviewed changes. Do not merge or rebase onto private full-suite history,
use `push --all`, push unrelated tags, or mirror the repository. Cherry-picking
applies selected changes without importing their original commit ancestry.

A version tag should point to the same reviewed public commit. Push only that tag,
then create a release on the public host if desired. Releasing does not filter
private content out of a branch. Check the included files, history, provider
examples, and chosen project license before the first public publication.
