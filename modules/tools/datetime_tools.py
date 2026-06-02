import calendar
from datetime import date as _Date, datetime as _DateTime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from modules.tools.registry import registry


@registry.register({
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": (
            "Get the current date and time. By default returns the device's local time. "
            "Optionally provide an IANA timezone name to get the time somewhere else. "
            "Use whenever the user asks what time or date it is, now or in another city."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": ["string", "null"],
                    "description": (
                        "IANA timezone name such as 'America/New_York', 'Europe/London', "
                        "or 'Asia/Tokyo'. Leave null for the device's local time."
                    ),
                },
            },
            "required": ["timezone"],
            "additionalProperties": False,
        },
    },
})
def get_current_time(timezone=None) -> str:
    if timezone:
        try:
            now = _DateTime.now(ZoneInfo(timezone))
            label = timezone
        except (ZoneInfoNotFoundError, ValueError):
            return (
                f"I don't recognize the timezone '{timezone}'. "
                "Try an IANA name like 'Asia/Tokyo' or 'Europe/Paris'."
            )
    else:
        now = _DateTime.now().astimezone()
        label = "local time"
    hour12 = now.strftime("%I").lstrip("0") or "12"
    return (
        f"It's {hour12}:{now.strftime('%M')} {now.strftime('%p')} on "
        f"{now.strftime('%A')}, {now.strftime('%B')} {now.day}, {now.year} ({label})."
    )


def _add_months(d: _Date, months: int) -> _Date:
    index = d.month - 1 + months
    year = d.year + index // 12
    month = index % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return _Date(year, month, day)


def _shift_date(d: _Date, amount: int, unit: str) -> _Date:
    if unit == "days":
        return d + timedelta(days=amount)
    if unit == "weeks":
        return d + timedelta(weeks=amount)
    if unit == "months":
        return _add_months(d, amount)
    if unit == "years":
        return _add_months(d, amount * 12)
    raise ValueError(f"unknown unit '{unit}'")


@registry.register({
    "type": "function",
    "function": {
        "name": "date_calculator",
        "description": (
            "Do date arithmetic. operation='difference' returns the number of days between "
            "two dates (use it for 'how many days until X' or 'days since Y'). operation='add' "
            "or 'subtract' shifts a date by an amount of days, weeks, months, or years and "
            "returns the resulting date. Dates are ISO format YYYY-MM-DD; a null date means today."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["difference", "add", "subtract"],
                    "description": "The kind of date calculation to perform.",
                },
                "date": {
                    "type": ["string", "null"],
                    "description": "Base date as YYYY-MM-DD. Null means today.",
                },
                "other_date": {
                    "type": ["string", "null"],
                    "description": "Second date as YYYY-MM-DD, only for operation='difference'. Null means today.",
                },
                "amount": {
                    "type": ["integer", "null"],
                    "description": "How many units to add or subtract, for operation='add'/'subtract'.",
                },
                "unit": {
                    "type": ["string", "null"],
                    "enum": ["days", "weeks", "months", "years", None],
                    "description": "Unit for add/subtract. Null defaults to days.",
                },
            },
            "required": ["operation", "date", "other_date", "amount", "unit"],
            "additionalProperties": False,
        },
    },
})
def date_calculator(operation, date=None, other_date=None, amount=None, unit=None) -> str:
    today = _Date.today()
    try:
        base = _DateTime.strptime(date, "%Y-%m-%d").date() if date else today
    except ValueError:
        return f"I couldn't read the date '{date}'. Please use YYYY-MM-DD format."

    if operation == "difference":
        try:
            other = _DateTime.strptime(other_date, "%Y-%m-%d").date() if other_date else today
        except ValueError:
            return f"I couldn't read the date '{other_date}'. Please use YYYY-MM-DD format."
        days = abs((other - base).days)
        if days == 0:
            return f"{base.isoformat()} and {other.isoformat()} are the same day."
        return f"There are {days} days between {base.isoformat()} and {other.isoformat()}."

    if operation in ("add", "subtract"):
        if amount is None:
            return "Tell me how many days, weeks, months, or years to add or subtract."
        unit = unit or "days"
        sign = 1 if operation == "add" else -1
        try:
            result = _shift_date(base, sign * int(amount), unit)
        except ValueError as e:
            return f"I couldn't do that: {e}."
        verb = "after" if operation == "add" else "before"
        return (
            f"{abs(int(amount))} {unit} {verb} {base.isoformat()} is "
            f"{result.isoformat()}, a {result.strftime('%A')}."
        )

    return f"Unknown operation '{operation}'. Use difference, add, or subtract."
