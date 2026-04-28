"""Parse pytest verbose output into per-test results."""

import re


def parse_log(log: str) -> dict[str, str]:
    """Parse test runner output into per-test results.

    Args:
        log: Full stdout+stderr output of `bash run_test.sh 2>&1`.

    Returns:
        Dict mapping test_id to status.
        - test_id: pytest native format e.g. "tests/foo.py::TestClass::test_func"
        - status: one of "PASSED", "FAILED", "SKIPPED", "ERROR"
    """
    results = {}
    # Strip ANSI color codes
    log = re.sub(r'\x1b\[[0-9;]*m', '', log)

    # Match verbose inline lines: "tests/foo.py::Bar::test_baz PASSED [ 42%]"
    # The test id may contain brackets for parametrized tests.
    inline_pattern = re.compile(
        r'^(tests/\S+::[\S ]+?)\s+(PASSED|FAILED|SKIPPED|ERROR)\s+\[\s*\d+%\]',
        re.MULTILINE
    )
    for m in inline_pattern.finditer(log):
        test_id = m.group(1).strip()
        status = m.group(2)
        results.setdefault(test_id, status)

    # Match summary section lines: "PASSED tests/foo.py::Bar::test_baz"
    summary_pattern = re.compile(
        r'^(PASSED|FAILED|SKIPPED|ERROR)\s+(tests/\S+)',
        re.MULTILINE
    )
    for m in summary_pattern.finditer(log):
        status = m.group(1)
        test_id = m.group(2).strip()
        results.setdefault(test_id, status)

    # Collect module-level ERROR entries (import failures): "ERROR tests/foo.py"
    error_pattern = re.compile(
        r'^ERROR\s+(tests/[^\s:]+\.py)\s*$',
        re.MULTILINE
    )
    for m in error_pattern.finditer(log):
        test_id = m.group(1).strip()
        results.setdefault(test_id, 'ERROR')

    return results


if __name__ == '__main__':
    log = open('test_output.log').read()
    results = parse_log(log)
    print(f"Parsed {len(results)} tests")
    print(f"Status distribution: { {s: sum(1 for v in results.values() if v == s) for s in set(results.values())} }")
    for test_id, status in sorted(results.items()):
        print(f"  {status:7s}  {test_id}")

