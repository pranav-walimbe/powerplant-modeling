# When to Mock

Avoid mocks by testing pure transformations with in-memory inputs. Do not mock
the filesystem, network, databases, subprocesses, or other boundaries because
those tests become integration-style coverage.

A tiny fake value is acceptable only when a pure unit accepts time, randomness,
or a callable dependency directly. Keep the fake local to one test and do not
assert call counts or internal collaboration details.
