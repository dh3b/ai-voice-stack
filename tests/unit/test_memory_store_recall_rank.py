"""
Runs against a throwaway SQLite file in tmp_path. Ranking is deterministic
(_relevance: +2 per key hit, +1 per value hit, +5 for an exact-phrase hit), so we can
assert exact orderings rather than vague "contains".
"""

from modules.utility.memory import MemoryStore


def _keys_in_order(recall_output):
    # recall() returns "- <key>: <value>" lines, best first.
    return [line.split(":", 1)[0].lstrip("- ").strip() for line in recall_output.splitlines()]


def test_memory_store_recall_rank(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))

    store.store("python_experience", "five years of backend work")
    store.store("hobbies", "I dabble in python and rust on weekends")
    store.store("trip", "visit Paris in the spring")
    store.store("chores", "spring cleaning, then a visit to the dentist")
    store.store("pet", "a dog named Rex")

    # Partial-word query: both rows match "pyth", but the key hit (+2) plus exact-phrase
    # bonus (+5) ranks python_experience above the value-only hit in hobbies.
    pyth = _keys_in_order(store.recall("pyth"))
    assert pyth[0] == "python_experience"
    assert "hobbies" in pyth

    # Exact-phrase hit outranks scattered single-word hits: "trip" contains the whole
    # phrase "visit paris"; "chores" only shares the word "visit".
    visit = _keys_in_order(store.recall("visit paris"))
    assert visit[0] == "trip"
    assert visit.index("trip") < visit.index("chores")

    # A miss surfaces everything currently stored, so the model can recover in one call.
    miss = store.recall("quantum chromodynamics")
    assert "Everything currently stored" in miss
    assert "python_experience" in miss

    # Delete, then confirm the fact is gone from recall.
    assert "Forgot 'pet'" in store.delete("pet")
    assert "No memory found with key 'pet'" in store.delete("pet")  # idempotent
    after = store.recall("rex")
    assert "Rex" not in after
