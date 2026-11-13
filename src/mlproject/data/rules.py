"""Data rules: every change to the data is a named function with a reason and a report (W2).

A rule never changes data silently. It returns the new data and a RuleReport, and the reports
of all rules go to MLflow as cleaning_log.json. Add your own rules below drop_above.
"""

from dataclasses import dataclass


@dataclass
class RuleReport:
    rule: str
    reason: str
    rows_before: int
    rows_after: int
    rows_flagged: int = 0  # rows a rule only marks or counts, without removing them


def drop_above(df, column, threshold, reason):
    """Removes rows with df[column] > threshold. Fit the threshold on the training part only."""
    out = df[df[column] <= threshold]
    return out, RuleReport(f"drop_above:{column}>{threshold:.2f}", reason, len(df), len(out))


def count_equal(df, column, value, reason):
    """Keeps all rows and reports how many have df[column] == value, e.g. a capped value."""
    flagged = int((df[column] == value).sum())
    return df, RuleReport(f"count_equal:{column}=={value}", reason, len(df), len(df), flagged)
