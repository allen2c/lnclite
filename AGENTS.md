# Agent guidance (Python)

- Prefer **pythonic**, **elegant** code with an explicit **declared** style: clear types, readable names, small focused units.
- Follow the Boy Scout Rule: leave touched code a little cleaner than you found it, while keeping cleanup tightly scoped to the task.
- Keep test files focused: one test function per test file. Put shared setup in fixtures or helpers instead of grouping multiple tests in one file.
- Order each module as follows (top to bottom):
  1. Module docstring
  2. Imports
  3. Constants
  4. Public functions
  5. Public classes
  6. Private helpers (functions / classes), if any
