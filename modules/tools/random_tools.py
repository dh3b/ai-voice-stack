import random

from modules.tools.registry import registry


@registry.register({
    "type": "function",
    "function": {
        "name": "random_generator",
        "description": (
            "Produce a random result. mode='number' picks a random integer between min and max "
            "(defaults 1 to 100); mode='dice' rolls dice (sides default 6, count default 1); "
            "mode='coin' flips a coin; mode='choice' picks one item from the options list. "
            "Use for 'roll a die', 'flip a coin', 'pick a number', or 'choose one of these'."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["number", "dice", "coin", "choice"],
                    "description": "Which kind of random result to produce.",
                },
                "min": {
                    "type": ["integer", "null"],
                    "description": "Lowest value for mode='number'. Defaults to 1.",
                },
                "max": {
                    "type": ["integer", "null"],
                    "description": "Highest value for mode='number'. Defaults to 100.",
                },
                "sides": {
                    "type": ["integer", "null"],
                    "minimum": 2,
                    "description": "Sides per die for mode='dice'. Defaults to 6.",
                },
                "count": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "description": "How many dice to roll for mode='dice'. Defaults to 1.",
                },
                "options": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Candidate items for mode='choice'.",
                },
            },
            "required": ["mode", "min", "max", "sides", "count", "options"],
            "additionalProperties": False,
        },
    },
})
def random_generator(mode, min=None, max=None, sides=None, count=None, options=None) -> str:
    mode = (mode or "").lower().strip()
    if mode == "number":
        lo = 1 if min is None else int(min)
        hi = 100 if max is None else int(max)
        if lo > hi:
            lo, hi = hi, lo
        return f"You got {random.randint(lo, hi)} (between {lo} and {hi})."
    if mode == "dice":
        faces = 6 if sides is None else int(sides)
        rolls_n = 1 if count is None else int(count)
        if faces < 2:
            return "A die needs at least 2 sides."
        if not 1 <= rolls_n <= 100:
            return "I can roll between 1 and 100 dice at once."
        rolls = [random.randint(1, faces) for _ in range(rolls_n)]
        if rolls_n == 1:
            return f"You rolled a {rolls[0]} on a {faces}-sided die."
        joined = ", ".join(str(r) for r in rolls)
        return f"You rolled {joined} on {rolls_n} {faces}-sided dice, for a total of {sum(rolls)}."
    if mode == "coin":
        return f"It's {random.choice(['heads', 'tails'])}."
    if mode == "choice":
        if not options:
            return "Give me a list of options to choose from."
        return f"I choose: {random.choice(list(options))}."
    return f"Unknown mode '{mode}'. Use number, dice, coin, or choice."
