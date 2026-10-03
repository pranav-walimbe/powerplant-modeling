---
name: tdd
description: Minimal local test-driven development with a red-green-refactor loop. Use only when the user explicitly asks for TDD, red-green-refactor, test-first development, or a focused unit test. Do not use for integration or end-to-end testing.
---

# Test-Driven Development

## Repository Constraints

- Unit tests are local verification artifacts under the ignored `/tests/` directory.
- Never force-add a test or modify ignore rules to place tests under Git tracking.
- Never add integration, end-to-end, snapshot, plotting, filesystem, network, subprocess, or artifact-producing tests.
- Prefer no new test when lint, type, compile, or a direct smoke command verifies the change adequately.
- Delete stale tests rather than preserving obsolete interfaces or behavior.
- Confirm every new test is ignored with `git check-ignore` and absent from `git status`.

## Philosophy

Write the smallest isolated unit test that proves the critical invariant. Avoid broad fixtures, real services, file writes, and incidental assertions. A test may exercise a private pure helper when that is the narrowest stable boundary.

See [tests.md](tests.md) for examples and [mocking.md](mocking.md) for mocking guidelines.

## Anti-Pattern: Horizontal Slices

**DO NOT write all tests first, then all implementation.** This is "horizontal slicing" - treating RED as "write all tests" and GREEN as "write all code."

This produces **crap tests**:

- Tests written in bulk test _imagined_ behavior, not _actual_ behavior
- You end up testing the _shape_ of things (data structures, function signatures) rather than user-facing behavior
- Tests become insensitive to real changes - they pass when behavior breaks, fail when behavior is fine
- You outrun your headlights, committing to test structure before understanding the implementation

**Correct approach**: Vertical slices via tracer bullets. One test → one implementation → repeat. Each test responds to what you learned from the previous cycle. Because you just wrote the code, you know exactly what behavior matters and how to verify it.

```
WRONG (horizontal):
  RED:   test1, test2, test3, test4, test5
  GREEN: impl1, impl2, impl3, impl4, impl5

RIGHT (vertical):
  RED→GREEN: test1→impl1
  RED→GREEN: test2→impl2
  RED→GREEN: test3→impl3
  ...
```

## Workflow

### 1. Planning

When exploring the codebase, use the project's domain glossary so that test names and interface vocabulary match the project's language, and respect ADRs in the area you're touching.

Before writing any code:

- [ ] Identify the single highest-risk behavior
- [ ] Confirm it can be tested without integration-style setup
- [ ] Confirm `/tests/` is ignored

Do not expand a code request into a test-suite project.

### 2. Tracer Bullet

Write ONE test that confirms ONE thing about the system:

```
RED:   Write test for first behavior → test fails
GREEN: Write minimal code to pass → test passes
```

This tracer bullet proves the narrow behavior before implementation expands.

### 3. Incremental Loop

Only if another critical behavior remains unverified:

```
RED:   Write next test → fails
GREEN: Minimal code to pass → passes
```

Rules:

- One test at a time
- Only enough code to pass current test
- Don't anticipate future tests
- Keep tests focused on observable behavior
- Stop as soon as the critical invariant is covered

### 4. Refactor

After all tests pass, look for [refactor candidates](refactoring.md):

- [ ] Extract duplication
- [ ] Deepen modules (move complexity behind simple interfaces)
- [ ] Apply SOLID principles where natural
- [ ] Consider what new code reveals about existing code
- [ ] Run tests after each refactor step

**Never refactor while RED.** Get to GREEN first.

## Checklist Per Cycle

```
[ ] Test describes behavior, not implementation
[ ] Test would survive internal refactor
[ ] Code is minimal for this test
[ ] No speculative features added
[ ] Test performs no integration-style I/O
[ ] Test remains ignored and untracked
```
