from perception_llm import LanguageMemoryDB

db = LanguageMemoryDB("language_memory.db")
n = db.migrate_scores()
print(f"Fixed {n} BLOB score rows")
db.close()
print("Done.")
