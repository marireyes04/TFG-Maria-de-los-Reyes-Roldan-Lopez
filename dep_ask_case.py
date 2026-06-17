from chat_profesional import init_backend, get_vectorstore
import json

print("Inicializando backend...")
init_backend()

vs = get_vectorstore()

raw = vs.get(
    where={"type": {"$eq": "case"}},
    include=["metadatas"]
)

metadatas = raw.get("metadatas") or []

ids = []
for meta in metadatas:
    if meta and "interview_id" in meta:
        ids.append(meta["interview_id"])

unique_ids = sorted(set(ids), key=str)

print(f"Total metadatas: {len(metadatas)}")
print(f"Unique interview_ids: {len(unique_ids)}")
print("Primeros 50 interview_id:")
for x in unique_ids[:50]:
    print(repr(x))